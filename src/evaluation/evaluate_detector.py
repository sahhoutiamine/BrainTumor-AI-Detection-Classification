
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


DATA = "data/processed/detection/data.yaml"
IMAGES = Path("data/processed/detection/images/test")
LABELS = Path("data/processed/detection/labels/test")
REPORTS = Path("reports/detection")


def calculer_iou(a, b):
    """Calculate IoU between two xyxy boxes."""

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])

    union = aire_a + aire_b - intersection
    return intersection / union if union > 0 else 0


def lire_labels(image):
    """Read real YOLO boxes and convert to pixels."""

    hauteur, largeur = cv2.imread(str(image)).shape[:2]
    fichier = LABELS / f"{image.stem}.txt"
    objets = []

    for ligne in fichier.read_text().splitlines():
        c, x, y, w, h = map(float, ligne.split())

        objets.append((
            int(c),
            [(x-w/2)*largeur, (y-h/2)*hauteur,
             (x+w/2)*largeur, (y+h/2)*hauteur]
        ))

    return objets


def evaluer():
    """Evaluate YOLO on the final test dataset."""

    REPORTS.mkdir(parents=True, exist_ok=True)
    modele = YOLO("models/detection/yolo_best.pt")

    metrics = modele.val(
        data=DATA,
        split="test",
        imgsz=640,
        device="cpu",
        plots=True,
        project=str(REPORTS),
        name="metrics",
        exist_ok=True
    )

    print("mAP50:", round(metrics.box.map50, 4))
    print("mAP50-95:", round(metrics.box.map, 4))
    print("Precision:", round(metrics.box.mp, 4))
    print("Recall:", round(metrics.box.mr, 4))

    tp = fp = fn = 0
    erreurs = []

    images = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    for image in images:
        resultat = modele.predict(
            str(image), conf=0.25, verbose=False
        )[0]

        vrais = lire_labels(image)
        utilises = set()
        faux_positifs = 0

        for box in resultat.boxes:
            classe = int(box.cls.item())
            coords = box.xyxy[0].tolist()
            meilleur = 0
            index = -1

            for i, (vraie_classe, vraie_box) in enumerate(vrais):
                if i in utilises or classe != vraie_classe:
                    continue

                iou = calculer_iou(coords, vraie_box)

                if iou > meilleur:
                    meilleur = iou
                    index = i

            if meilleur >= 0.5:
                tp += 1
                utilises.add(index)
            else:
                fp += 1
                faux_positifs += 1

        manques = len(vrais) - len(utilises)
        fn += manques

        if faux_positifs or manques:
            erreurs.append({
                "image": image.name,
                "false_positives": faux_positifs,
                "missed_tumors": manques
            })

            if len(erreurs) <= 10:
                cv2.imwrite(
                    str(REPORTS / image.name),
                    resultat.plot()
                )

    pd.DataFrame(
        erreurs,
        columns=["image", "false_positives", "missed_tumors"]
    ).to_csv(REPORTS / "errors.csv", index=False)

    print("\nIoU threshold: 0.5")
    print("True Positives:", tp)
    print("False Positives:", fp)
    print("Missed Tumors:", fn)
    print("Detection results saved in:", REPORTS)


if __name__ == "__main__":
    evaluer()
