
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data/raw")
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def lire_boites(label_path, classe):
    """Read and validate YOLO annotations."""
    if not label_path.exists():
        if classe == 2:
            return []
        raise ValueError("Annotation manquante")

    boites = []

    for ligne in label_path.read_text().splitlines():
        valeurs = ligne.split()

        if len(valeurs) != 5:
            raise ValueError("Format YOLO incorrect")

        nombres = [float(v) for v in valeurs]
        id_classe, x, y, w, h = nombres

        if id_classe not in [0, 1, 2, 3]:
            raise ValueError("Classe inconnue")

        if not (
            0 <= x <= 1 and 0 <= y <= 1
            and w > 0 and h > 0
            and x - w / 2 >= -0.00001
            and y - h / 2 >= -0.00001
            and x + w / 2 <= 1.00001
            and y + h / 2 <= 1.00001
        ):
            raise ValueError("Bounding box incorrecte")

        boites.append(nombres)

    # No Tumor has no real tumor to detect.
    # Ignore artificial boxes for detection.
    if classe == 2:
        return []

    if not boites:
        raise ValueError("Aucune boite tumorale")

    if any(int(b[0]) != classe for b in boites):
        raise ValueError("Conflit entre classes")

    return boites


def explorer():
    """Inspect all images and save an audit."""
    resultats = []
    compteurs = Counter()

    for partie in ["Train", "Val"]:
        for classe, nom in enumerate(CLASSES):
            dossier = ROOT / partie / nom

            for image_path in sorted(
                (dossier / "images").glob("*")
            ):
                if image_path.suffix.lower() not in EXTENSIONS:
                    continue

                probleme = ""
                boites = []
                label = dossier / "labels" / f"{image_path.stem}.txt"

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    boites = lire_boites(label, classe)

                except Exception as erreur:
                    probleme = str(erreur)

                resultats.append({
                    "image": str(image_path),
                    "classe": classe,
                    "partie": partie,
                    "nb_boites": len(boites),
                    "probleme": probleme,
                })

                compteurs[nom] += 1

    df = pd.DataFrame(resultats)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/audit.csv", index=False)

    print("Total images:", len(df))
    print("Distribution:", dict(compteurs))
    print("Images problématiques:",
          (df["probleme"] != "").sum())

    plt.bar(compteurs.keys(), compteurs.values())
    plt.title("Distribution des classes")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/distribution.png")
    plt.close()

    return df


def visualiser():
    """Show one annotated example for each class."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for classe, nom in enumerate(CLASSES):
        dossier = ROOT / "Train" / nom
        images = list((dossier / "images").glob("*"))
        images = [p for p in images if p.suffix.lower() in EXTENSIONS]

        if not images:
            continue

        image_path = images[0]
        label = dossier / "labels" / f"{image_path.stem}.txt"

        with Image.open(image_path) as fichier:
            img = fichier.convert("RGB")

        axes[classe].imshow(img)
        axes[classe].set_title(nom)
        axes[classe].axis("off")

        try:
            boites = lire_boites(label, classe)
        except (ValueError, OSError):
            boites = []

        largeur, hauteur = img.size

        for _, x, y, w, h in boites:
            rectangle = plt.Rectangle(
                ((x - w/2)*largeur, (y - h/2)*hauteur),
                w*largeur,
                h*hauteur,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
            axes[classe].add_patch(rectangle)

    plt.tight_layout()
    plt.savefig("reports/exemples.png")
    plt.close()


if __name__ == "__main__":
    explorer()
    visualiser()
