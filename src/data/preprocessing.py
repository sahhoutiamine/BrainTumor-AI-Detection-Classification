
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
SIZE = (224, 224)
BATCH_SIZE = 32

tf.keras.utils.set_random_seed(SEED)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


def lire_image(chemin, label):
    """Read, resize and normalize an MRI."""
    image = tf.io.read_file(chemin)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augmenter(image, label):
    """Apply random changes only during training."""
    image = augmentation(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def adapter_modele(image, label, modele):
    """Use the right ImageNet preprocessing."""
    if modele == "MobileNetV2":
        image = image * 2.0 - 1.0

    elif modele == "ResNet50":
        image = tf.keras.applications.resnet50.preprocess_input(
            image * 255.0
        )

    return image, label


def charger_dataset(partie, modele="MobileNetV2"):
    """Create a TensorFlow dataset from a CSV."""
    df = pd.read_csv(f"data/processed/{partie}.csv")

    ds = tf.data.Dataset.from_tensor_slices((
        df["image"].values,
        df["classe"].values.astype("int32")
    ))

    if partie == "train":
        ds = ds.shuffle(
            len(df),
            seed=SEED
        )

    ds = ds.map(
        lire_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if partie == "train":
        ds = ds.map(
            augmenter,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.map(
        lambda x, y: adapter_modele(x, y, modele),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )


def augmenter_detection(image_path, label_path):
    """Augment an image and its YOLO bounding boxes."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image illisible: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boites = []
    classes = []

    for ligne in Path(label_path).read_text().splitlines():
        valeurs = ligne.split()
        classes.append(int(valeurs[0]))
        boites.append([float(v) for v in valeurs[1:]])

    transformation = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["classes"],
            min_visibility=0.25
        )
    )

    resultat = transformation(
        image=image,
        bboxes=boites,
        classes=classes
    )

    return (
        resultat["image"],
        resultat["bboxes"],
        resultat["classes"]
    )


if __name__ == "__main__":
    for partie in ["train", "val", "test"]:
        ds = charger_dataset(partie)
        images, labels = next(iter(ds))

        print(
            partie,
            "images:", images.shape,
            "labels:", labels.shape,
            "minimum:", float(tf.reduce_min(images)),
            "maximum:", float(tf.reduce_max(images))
        )
