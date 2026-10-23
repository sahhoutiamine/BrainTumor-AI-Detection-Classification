
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


# -------------------------------------
# Configuration
# -------------------------------------

RANDOM_STATE = 42

CLASSIFICATION_SIZE = 224
DETECTION_SIZE = 640

DETECTION_CLASSES = [
    "Glioma",
    "Meningioma",
    "Pituitary",
]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# -------------------------------------
# Classification augmentation
# -------------------------------------

def get_classification_augmentation():
    """
    Apply moderate random transformations
    only to training images.

    Input:
        RGB image (224 x 224 x 3)

    Output:
        Augmented RGB image
    """

    transform = A.Compose(
        [
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.4,
            ),
        ]
    )

    return transform


# -------------------------------------
# Detection augmentation
# -------------------------------------

def get_detection_augmentation():
    """
    Augment images and their bounding boxes
    together.

    Bounding boxes must be in YOLO format:

    [x_center, y_center, width, height]

    All values must be normalized [0, 1].
    """

    transform = A.Compose(
        [
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.4,
            ),
        ],

        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )

    return transform


# -------------------------------------
# Augment classification image
# -------------------------------------

def augment_classification_image(image):
    """
    Receives an RGB image and returns
    its augmented version.
    """

    transform = get_classification_augmentation()

    result = transform(image=image)

    return result["image"]


# -------------------------------------
# Augment detection image + boxes
# -------------------------------------

def augment_detection_image(
    image,
    bboxes,
    class_labels,
):
    """
    Apply the SAME transformations to
    an image and its bounding boxes.

    Args:
        image: RGB NumPy image
        bboxes: list of YOLO bounding boxes
        class_labels: class ID for each box

    Returns:
        augmented_image
        augmented_boxes
        augmented_labels
    """

    if len(bboxes) != len(class_labels):
        raise ValueError(
            "Each bounding box needs a class label."
        )

    transform = get_detection_augmentation()

    result = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )

    augmented_image = result["image"]

    augmented_boxes = list(result["bboxes"])

    augmented_labels = list(
        result["class_labels"]
    )

    return (
        augmented_image,
        augmented_boxes,
        augmented_labels,
    )


# -------------------------------------
# Read YOLO annotation
# -------------------------------------

def read_yolo_annotation(label_path):
    """
    Read a YOLO .txt annotation file.

    Example:
        0 0.5 0.4 0.2 0.3

    For No Tumor images, an empty
    annotation file is valid.
    """

    bboxes = []
    class_labels = []

    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(
            f"Annotation not found: {label_path}"
        )

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation at "
                    f"{label_path}, line {line_number}"
                )

            class_value = float(values[0])

            if not class_value.is_integer():
                raise ValueError(
                    "YOLO class IDs must be integers."
                )

            class_id = int(class_value)

            box = [
                float(value)
                for value in values[1:]
            ]

            if class_id not in range(
                len(DETECTION_CLASSES)
            ):
                raise ValueError(
                    f"Unknown detection class: {class_id}"
                )

            x, y, width, height = box

            if not all(
                np.isfinite(value)
                for value in box
            ):
                raise ValueError(
                    "Non-finite bounding-box coordinate."
                )

            if (
                width <= 0
                or height <= 0
                or x - width / 2 < -1e-6
                or y - height / 2 < -1e-6
                or x + width / 2 > 1 + 1e-6
                or y + height / 2 > 1 + 1e-6
            ):
                raise ValueError(
                    f"Invalid bounding box: {box}"
                )

            bboxes.append(box)
            class_labels.append(class_id)

    return bboxes, class_labels


# -------------------------------------
# Draw bounding boxes
# -------------------------------------

def draw_bounding_boxes(
    image,
    bboxes,
    class_labels,
):
    """
    Draw YOLO bounding boxes
    on an RGB image.
    """

    result = image.copy()

    height, width = image.shape[:2]

    for box, class_id in zip(
        bboxes,
        class_labels,
    ):

        x, y, box_w, box_h = box

        x1 = int((x - box_w / 2) * width)
        y1 = int((y - box_h / 2) * height)

        x2 = int((x + box_w / 2) * width)
        y2 = int((y + box_h / 2) * height)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        label = DETECTION_CLASSES[int(class_id)]

        cv2.putText(
            result,
            label,
            (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    return result


# -------------------------------------
# Preview classification augmentations
# -------------------------------------

def preview_classification(image_path):
    """
    Generate a preview with the original
    and five randomly augmented versions.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot open image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (
            CLASSIFICATION_SIZE,
            CLASSIFICATION_SIZE,
        ),
    )

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    for i in range(5):

        augmented = augment_classification_image(
            image
        )

        plt.subplot(2, 3, i + 2)
        plt.imshow(augmented)
        plt.title(f"Augmentation {i + 1}")
        plt.axis("off")

    output_dir = Path("reports/figures")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "classification_augmentation.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()

    print(f"Saved: {output_path}")


# -------------------------------------
# Preview detection augmentations
# -------------------------------------

def preview_detection(
    image_path,
    label_path,
):
    """
    Visualize original and augmented
    images with bounding boxes.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot open image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    bboxes, labels = read_yolo_annotation(
        label_path
    )

    original = draw_bounding_boxes(
        image,
        bboxes,
        labels,
    )

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    for i in range(5):

        aug_image, aug_boxes, aug_labels = (
            augment_detection_image(
                image,
                bboxes,
                labels,
            )
        )

        visualization = draw_bounding_boxes(
            aug_image,
            aug_boxes,
            aug_labels,
        )

        plt.subplot(2, 3, i + 2)
        plt.imshow(visualization)
        plt.title(f"Augmentation {i + 1}")
        plt.axis("off")

    output_dir = Path("reports/figures")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "detection_augmentation.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()

    print(f"Saved: {output_path}")


# -------------------------------------
# Run previews from the terminal
# -------------------------------------

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["classification", "detection"],
        required=True,
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--label",
        default=None,
    )

    args = parser.parse_args()

    if args.task == "classification":

        preview_classification(
            args.image
        )

    else:

        if args.label is None:
            parser.error(
                "--label is required for detection."
            )

        preview_detection(
            args.image,
            args.label,
        )
