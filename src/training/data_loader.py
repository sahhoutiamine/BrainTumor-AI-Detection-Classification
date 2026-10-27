
import numpy as np
import tensorflow as tf

from src.data.preprocessing import (
    MRIDataGenerator,
    get_image_paths,
)


class ResNetDataGenerator(MRIDataGenerator):
    """
    Reuse the existing MRI generator.

    Convert MobileNetV2-normalized images
    back to RGB values in [0, 255].

    Then apply ResNet50 preprocessing.
    """

    def __getitem__(self, index):

        images, labels = super().__getitem__(
            index
        )

        images = (
            images + 1.0
        ) * 127.5

        images = images.astype(
            np.float32
        )

        images = (
            tf.keras.applications.resnet50
            .preprocess_input(images)
        )

        return images, labels


def create_model_datasets(
    model_name,
    batch_size=32,
):
    """
    Create the datasets with the correct
    preprocessing for each architecture.
    """

    model_name = model_name.lower()

    if model_name == "mobilenetv2":

        generator_class = MRIDataGenerator

    elif model_name == "resnet50":

        generator_class = ResNetDataGenerator

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    train_paths, train_labels = get_image_paths(
        "train"
    )

    val_paths, val_labels = get_image_paths(
        "val"
    )

    test_paths, test_labels = get_image_paths(
        "test"
    )

    train_data = generator_class(
        train_paths,
        train_labels,
        batch_size=batch_size,
        training=True,
    )

    val_data = generator_class(
        val_paths,
        val_labels,
        batch_size=batch_size,
        training=False,
    )

    test_data = generator_class(
        test_paths,
        test_labels,
        batch_size=batch_size,
        training=False,
    )

    print(
        f"\nArchitecture: {model_name}"
    )

    print(
        f"Training images: {len(train_paths)}"
    )

    print(
        f"Validation images: {len(val_paths)}"
    )

    print(
        f"Testing images: {len(test_paths)}"
    )

    return train_data, val_data, test_data
