
from pathlib import Path
from collections import Counter
from hashlib import sha256
from shutil import copy2


import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.exploration import explorer, lire_boites, CLASSES


SORTIE = Path("data/processed")
DETECTION = SORTIE / "detection"


def supprimer_doublons(df):
    """Keep one copy of each exact image."""
    empreintes = set()
    gardes = []
    supp = []

    # Give the original Val partition priority
    # because we reserve it for the final test.
    df = df.sort_values(
        "partie",
        key=lambda x: x.map({"Val": 0, "Train": 1})
    )

    for _, ligne in df.iterrows():
        contenu = Path(ligne["image"]).read_bytes()
        empreinte = sha256(contenu).hexdigest()

        if empreinte not in empreintes:
            empreintes.add(empreinte)
            gardes.append(ligne.to_dict())
        else :
            supp.append(ligne.to_dict())
            
    print(f"Supprimés {len(supp)} doublons sur {len(df)} images"
)
    return pd.DataFrame(gardes)


def exporter_detection(df, partie):
    """Copy images and convert labels for YOLO."""
    images_dir = DETECTION / "images" / partie
    labels_dir = DETECTION / "labels" / partie

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Classification IDs -> detection IDs
    mapping = {0: 0, 1: 1, 3: 2}

    for numero, (_, ligne) in enumerate(df.iterrows()):
        image = Path(ligne["image"])
        classe = int(ligne["classe"])

        nom = f"{partie}_{numero:05d}"
        destination = images_dir / f"{nom}{image.suffix.lower()}"

        copy2(image, destination)

        source_label = (
            image.parent.parent / "labels" / f"{image.stem}.txt"
        )

        boites = lire_boites(source_label, classe)
        lignes = []

        for id_classe, x, y, w, h in boites:
            nouvel_id = mapping[int(id_classe)]
            lignes.append(
                f"{nouvel_id} {x} {y} {w} {h}"
            )

        (labels_dir / f"{nom}.txt").write_text(
            "\n".join(lignes)
        )


def preparer():
    """Create train, validation and test datasets."""
    df = explorer()

    df = df[df["probleme"] == ""].copy()
    df = supprimer_doublons(df)

    origine_train = df[df["partie"] == "Train"]
    test = df[df["partie"] == "Val"]

    train, val = train_test_split(
        origine_train,
        test_size=0.15,
        random_state=42,
        stratify=origine_train["classe"]
    )

    SORTIE.mkdir(parents=True, exist_ok=True)

    for nom, donnees in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        donnees[["image", "classe"]].to_csv(
            SORTIE / f"{nom}.csv",
            index=False
        )

        exporter_detection(donnees, nom)

        print(f"{nom}: {len(donnees)} images")
        print(dict(Counter(donnees["classe"])))

    config = {
        "path": str(DETECTION.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "Glioma",
            1: "Meningioma",
            2: "Pituitary"
        }
    }

    with open(DETECTION / "data.yaml", "w") as fichier:
        yaml.safe_dump(config, fichier, sort_keys=False)

    print("Préparation terminée !")


if __name__ == "__main__":
    preparer()
