
from pathlib import Path
from ultralytics import YOLO
import yaml


DATA = Path("data/processed/detection/data.yaml")
MODELE = Path("models/detection/yolo_best.pt")
RESULTATS = Path("reports/yolo")

CLASSES = ["Glioma", "Meningioma", "Pituitary"]


def verifier_annotations():
    """Check the YOLO dataset before training."""

    with open(DATA, "r") as f:
        config = yaml.safe_load(f)

    dossier = Path(config["path"])

    assert list(config["names"].values()) == CLASSES

    for partie in ["train", "val", "test"]:
        images = list((dossier / "images" / partie).glob("*"))
        labels = dossier / "labels" / partie

        assert images, f"No images in {partie}"

        for image in images:
            label = labels / f"{image.stem}.txt"
            assert label.exists(), f"Missing label: {label}"

            for ligne in label.read_text().splitlines():
                valeurs = ligne.split()
                assert len(valeurs) == 5, f"Invalid: {label}"

                classe = int(valeurs[0])
                x, y, w, h = map(float, valeurs[1:])

                assert classe in [0, 1, 2]
                assert w > 0 and h > 0
                assert x - w / 2 >= -0.00001
                assert y - h / 2 >= -0.00001
                assert x + w / 2 <= 1.00001
                assert y + h / 2 <= 1.00001

        print(f"{partie}: {len(images)} images verified")


def entrainer():
    """Train a pretrained YOLOv8 model."""

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA),
        epochs=50,       
        patience=10,     
        batch=32,
        imgsz=640,
        device="cpu",

        # Moderate MRI augmentation
        degrees=10,
        translate=0.05,
        scale=0.10,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        mosaic=0.0,

        project=str(RESULTATS),
        name="training",
        exist_ok=True,
        plots=True
    )

    MODELE.parent.mkdir(parents=True, exist_ok=True)

    meilleur = Path(model.trainer.best)
    MODELE.write_bytes(meilleur.read_bytes())

    print("Best model saved:", MODELE)


def evaluer():
    """Evaluate the detector on validation data."""

    model = YOLO(str(MODELE))

    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        device="cpu",
        plots=True,
        project=str(RESULTATS),
        name="validation",
        exist_ok=True
    )

    print("Precision:", round(metrics.box.mp, 4))
    print("Recall:", round(metrics.box.mr, 4))
    print("mAP50:", round(metrics.box.map50, 4))
    print("mAP50-95:", round(metrics.box.map, 4))


def predire():
    """Save predicted bounding boxes on sample MRIs."""

    model = YOLO(str(MODELE))

    images = sorted(
        Path("data/processed/detection/images/val").glob("*")
    )

    model.predict(
        source=[str(image) for image in images[:12]],
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(RESULTATS),
        name="predictions",
        exist_ok=True
    )

    print("Predictions saved in:", RESULTATS / "predictions")


if __name__ == "__main__":
    verifier_annotations()
    entrainer()
    evaluer()
    predire()
