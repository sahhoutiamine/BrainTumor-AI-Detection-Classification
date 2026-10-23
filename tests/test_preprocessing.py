
import cv2
import numpy as np
import pytest

from src.data.preprocessing import (
    IMAGE_SIZE,
    load_image,
    resize_image,
    normalize_image,
    preprocess_image,
)

from src.data.augmentation import (
    augment_classification_image,
    augment_detection_image,
    read_yolo_annotation,
)


# -------------------------------------
# Create a fake MRI image for testing
# -------------------------------------

@pytest.fixture
def sample_image():
    """
    Create a synthetic image.

    We don't need the real dataset
    for these unit tests.
    """

    image = np.zeros(
        (300, 400, 3),
        dtype=np.uint8,
    )

    cv2.circle(
        image,
        (200, 150),
        60,
        (180, 180, 180),
        -1,
    )

    return image


# -------------------------------------
# Test resizing
# -------------------------------------

def test_resize_image(sample_image):

    resized = resize_image(
        sample_image
    )

    assert resized.shape == (
        224,
        224,
        3,
    )


# -------------------------------------
# Test normalization
# -------------------------------------

def test_normalize_image():

    image = np.array(
        [[[0, 127, 255]]],
        dtype=np.uint8,
    )

    normalized = normalize_image(
        image
    )

    assert normalized.dtype == np.float32

    assert normalized.min() >= -1.0

    assert normalized.max() <= 1.0

    assert np.isclose(
        normalized[0, 0, 0],
        -1.0,
    )

    assert np.isclose(
        normalized[0, 0, 2],
        1.0,
    )


# -------------------------------------
# Test classification augmentation
# -------------------------------------

def test_classification_augmentation(
    sample_image,
):

    resized = resize_image(
        sample_image
    )

    augmented = (
        augment_classification_image(
            resized
        )
    )

    assert augmented.shape == (
        224,
        224,
        3,
    )

    assert augmented.dtype == np.uint8


# -------------------------------------
# Test detection augmentation
# -------------------------------------

def test_detection_augmentation(
    sample_image,
):

    bboxes = [
        [0.5, 0.5, 0.3, 0.3]
    ]

    labels = [0]

    (
        augmented_image,
        augmented_boxes,
        augmented_labels,
    ) = augment_detection_image(
        sample_image,
        bboxes,
        labels,
    )

    assert augmented_image.shape == (
        300,
        400,
        3,
    )

    assert len(augmented_boxes) == len(
        augmented_labels
    )

    for box in augmented_boxes:

        x, y, width, height = box

        assert 0 <= x <= 1
        assert 0 <= y <= 1

        assert 0 < width <= 1
        assert 0 < height <= 1

        assert x - width / 2 >= -1e-5
        assert y - height / 2 >= -1e-5

        assert x + width / 2 <= 1 + 1e-5
        assert y + height / 2 <= 1 + 1e-5


# -------------------------------------
# Test YOLO annotation reading
# -------------------------------------

def test_read_yolo_annotation(
    tmp_path,
):

    label_file = (
        tmp_path / "example.txt"
    )

    label_file.write_text(
        "0 0.5 0.5 0.3 0.3\n",
        encoding="utf-8",
    )

    boxes, labels = read_yolo_annotation(
        label_file
    )

    assert len(boxes) == 1

    assert labels == [0]

    assert boxes[0] == [
        0.5,
        0.5,
        0.3,
        0.3,
    ]


# -------------------------------------
# Test No Tumor annotations
# -------------------------------------

def test_empty_detection_annotation(
    tmp_path,
):

    label_file = (
        tmp_path / "no_tumor.txt"
    )

    label_file.write_text(
        "",
        encoding="utf-8",
    )

    boxes, labels = read_yolo_annotation(
        label_file
    )

    assert boxes == []
    assert labels == []


# -------------------------------------
# Test invalid YOLO box
# -------------------------------------

def test_invalid_box(
    tmp_path,
):

    label_file = (
        tmp_path / "invalid.txt"
    )

    label_file.write_text(
        "0 0.95 0.5 0.4 0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        read_yolo_annotation(
            label_file
        )


# -------------------------------------
# Test preprocessing without augmentation
# -------------------------------------

def test_deterministic_preprocessing(
    tmp_path,
    sample_image,
):

    image_path = (
        tmp_path / "test.jpg"
    )

    cv2.imwrite(
        str(image_path),
        sample_image,
    )

    first = preprocess_image(
        image_path,
        training=False,
    )

    second = preprocess_image(
        image_path,
        training=False,
    )

    assert first.shape == (
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        3,
    )

    assert np.array_equal(
        first,
        second,
    )


# -------------------------------------
# Test image reading
# -------------------------------------

def test_load_image(
    tmp_path,
    sample_image,
):

    image_path = (
        tmp_path / "image.png"
    )

    cv2.imwrite(
        str(image_path),
        sample_image,
    )

    loaded = load_image(
        image_path
    )

    assert loaded.shape == (
        300,
        400,
        3,
    )


# -------------------------------------
# Test missing image
# -------------------------------------

def test_missing_image(tmp_path):

    missing_path = (
        tmp_path / "missing.png"
    )

    with pytest.raises(ValueError):

        load_image(
            missing_path
        )
