
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from src.data.preprocessing import charger_dataset


CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
Path("reports").mkdir(exist_ok=True)


def evaluer():
    """Evaluate the best classification model on test data."""

    # 1. Select the best model using validation F1
    df = pd.read_csv("reports/comparaison_step4.csv")
    meilleur = df.sort_values("f1_macro", ascending=False).iloc[0]

    nom = meilleur["modele"]
    configuration = meilleur["configuration"]
    phase = meilleur["phase"].lower().replace("-", "")

    chemin = (
        f"models/classification/"
        f"{nom}_{configuration}_{phase}.keras"
    )

    print("Best model:", chemin)

    # 2. Load the model and test dataset
    modele = tf.keras.models.load_model(chemin)
    test_ds = charger_dataset("test", nom)

    vrais = []
    predictions = []

    for images, labels in test_ds:
        resultat = modele(images, training=False)

        vrais.extend(labels.numpy())
        predictions.extend(tf.argmax(resultat, axis=1).numpy())

    # 3. Accuracy, Precision, Recall and F1-score
    rapport = classification_report(
        vrais,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    print("\nClassification results:")
    print(pd.DataFrame(rapport).transpose().round(3))

    pd.DataFrame(rapport).transpose().to_csv(
        "reports/classification_metrics.csv"
    )

    # 4. Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        vrais,
        predictions,
        labels=[0, 1, 2, 3],
        display_labels=CLASSES,
        cmap="Blues"
    )

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reports/classification_confusion.png")
    plt.close()

    # 5. Save incorrect predictions
    erreurs = pd.DataFrame({
        "image": pd.read_csv("data/processed/test.csv")["image"],
        "vraie_classe": [CLASSES[int(x)] for x in vrais],
        "prediction": [CLASSES[int(x)] for x in predictions]
    })

    erreurs = erreurs[
        erreurs["vraie_classe"] != erreurs["prediction"]
    ]

    erreurs.to_csv(
        "reports/classification_errors.csv",
        index=False
    )

    print("\nIncorrect predictions:", len(erreurs))
    print("Results saved in reports/")


if __name__ == "__main__":
    evaluer()
