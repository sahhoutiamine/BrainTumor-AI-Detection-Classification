
from pathlib import Path
import argparse
import math

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
)

from src.data.augmentation import (
    augment_classification_image,
)


# -------------------------------------
# Configuration
# -------------------------------------

DATA_DIR = Path(
    "data/processed/classification"
)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# -------------------------------------
# Load a single image
# -------------------------------------

def load_image(image_path):
    """
    Read an image using OpenCV.

    Returns:
        RGB image as NumPy array.
    """

    image_path = Path(image_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return image


# -------------------------------------
# Resize an image
# -------------------------------------

def resize_image(
    image,
    size=IMAGE_SIZE,
):
    """
    Resize an image to the given
    (width, height).
    """

    resized = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA,
    )

    return resized


# -------------------------------------
# Normalize an image
# -------------------------------------

def normalize_image(image):
    """
    Normalize an image for MobileNetV2.

    Original values:
        0 to 255

    New values:
        -1 to 1
    """

    image = image.astype(np.float32)

    normalized = preprocess_input(image)

    return normalized


# -------------------------------------
# Complete image preprocessing
# -------------------------------------

def preprocess_image(
    image_path,
    training=False,
):
    """
    Preprocessing pipeline:

    1. Read image
    2. Resize image
    3. Apply augmentation if training
    4. Normalize for MobileNetV2

    Returns:
        Shape (224, 224, 3)
        dtype float32
    """

    image = load_image(image_path)

    image = resize_image(image)

    if training:
        image = augment_classification_image(
            image
        )

    image = normalize_image(image)

    return image


# -------------------------------------
# Discover classification images
# -------------------------------------

def get_image_paths(split):
    """
    Read images and labels from:

    data/processed/classification/
        train/
        val/
        test/

    Each split must contain a folder
    for every class.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unknown dataset split: {split}"
        )

    split_dir = DATA_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_dir}"
        )

    image_paths = []
    labels = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        for image_path in sorted(
            class_dir.rglob("*")
        ):

            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            image_paths.append(image_path)
            labels.append(class_id)

    return image_paths, labels


# -------------------------------------
# Create a batch generator
# -------------------------------------

class MRIDataGenerator(tf.keras.utils.Sequence):
    """
    Loads MRI images in small batches.

    Training:
        Random augmentation + normalization

    Validation / Test:
        Normalization without augmentation
    """

    def __init__(
        self,
        image_paths,
        labels,
        batch_size=BATCH_SIZE,
        training=False,
        seed=RANDOM_STATE,
        **kwargs,
    ):

        super().__init__(**kwargs)

        if len(image_paths) != len(labels):
            raise ValueError(
                "Images and labels must have "
                "the same length."
            )

        if not image_paths:
            raise ValueError(
                "The dataset is empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self.image_paths = list(image_paths)

        self.labels = np.array(
            labels,
            dtype=np.int32,
        )

        self.batch_size = batch_size

        self.training = training

        self.rng = np.random.default_rng(
            seed
        )

        self.indices = np.arange(
            len(self.image_paths)
        )

        self.on_epoch_end()

    def __len__(self):
        """
        Number of batches per epoch.
        """

        return math.ceil(
            len(self.image_paths)
            / self.batch_size
        )

    def __getitem__(self, index):
        """
        Return one batch:

        X: (batch, 224, 224, 3)
        y: (batch,)
        """

        start = index * self.batch_size

        end = start + self.batch_size

        batch_indices = self.indices[
            start:end
        ]

        images = []
        labels = []

        for image_index in batch_indices:

            image_path = self.image_paths[
                image_index
            ]

            label = self.labels[
                image_index
            ]

            image = preprocess_image(
                image_path,
                training=self.training,
            )

            images.append(image)

            labels.append(label)

        X = np.stack(
            images
        ).astype(np.float32)

        y = np.array(
            labels,
            dtype=np.int32,
        )

        return X, y

    def on_epoch_end(self):
        """
        Shuffle the training data after
        every epoch.

        Never shuffle validation/test.
        """

        if self.training:

            self.rng.shuffle(
                self.indices
            )


# -------------------------------------
# Build the three datasets
# -------------------------------------

def create_datasets(
    batch_size=BATCH_SIZE,
):
    """
    Create:

        train_generator
        val_generator
        test_generator
    """

    train_paths, train_labels = (
        get_image_paths("train")
    )

    val_paths, val_labels = (
        get_image_paths("val")
    )

    test_paths, test_labels = (
        get_image_paths("test")
    )

    train_generator = MRIDataGenerator(
        train_paths,
        train_labels,
        batch_size=batch_size,
        training=True,
    )

    val_generator = MRIDataGenerator(
        val_paths,
        val_labels,
        batch_size=batch_size,
        training=False,
    )

    test_generator = MRIDataGenerator(
        test_paths,
        test_labels,
        batch_size=batch_size,
        training=False,
    )

    print("\nDataset information")
    print("-" * 40)

    print(
        f"Training images: {len(train_paths)}"
    )

    print(
        f"Validation images: {len(val_paths)}"
    )

    print(
        f"Testing images: {len(test_paths)}"
    )

    print(
        f"Training batches: {len(train_generator)}"
    )

    print(
        f"Validation batches: {len(val_generator)}"
    )

    print(
        f"Testing batches: {len(test_generator)}"
    )

    return (
        train_generator,
        val_generator,
        test_generator,
    )


# -------------------------------------
# Inspect the first batch
# -------------------------------------

def inspect_batch(generator):
    """
    Inspect images, labels, shape
    and normalization values.
    """

    X, y = generator[0]

    print("\nFirst batch")
    print("-" * 40)

    print("Image shape:", X.shape)

    print("Label shape:", y.shape)

    print("Image dtype:", X.dtype)

    print("Minimum pixel value:", X.min())

    print("Maximum pixel value:", X.max())

    print("Labels:", y.tolist())


# -------------------------------------
# Preview images from a batch
# -------------------------------------

def save_batch_preview(generator):
    """
    Save a preview of the first
    six preprocessed images.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    X, y = generator[0]

    num_images = min(6, len(X))

    plt.figure(
        figsize=(12, 7)
    )

    for i in range(num_images):

        # Convert MobileNetV2 values
        # from [-1, 1] back to [0, 1]
        # for visualization.

        image = (
            X[i] + 1.0
        ) / 2.0

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        plt.subplot(
            2,
            3,
            i + 1,
        )

        plt.imshow(image)

        plt.title(
            CLASS_NAMES[int(y[i])]
        )

        plt.axis("off")

    output_dir = Path(
        "reports/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "preprocessed_batch.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"\nBatch preview saved: {output_path}"
    )


# -------------------------------------
# Execute preprocessing
# -------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    train, val, test = create_datasets(
        batch_size=args.batch_size
    )

    print("\nTRAIN")
    inspect_batch(train)

    print("\nVALIDATION")
    inspect_batch(val)

    print("\nTEST")
    inspect_batch(test)

    save_batch_preview(train)

    print(
        "\nStep 2 preprocessing completed."
    )


if __name__ == "__main__":
    main()
