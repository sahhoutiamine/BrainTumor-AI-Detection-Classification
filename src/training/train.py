
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import f1_score
from src.data.preprocessing import charger_dataset


SEED = 42
CLASSES = 4
DOSSIER = Path("models/classification")
RAPPORTS = Path("reports")

DOSSIER.mkdir(parents=True, exist_ok=True)
RAPPORTS.mkdir(exist_ok=True)


def creer_modele(nom, dropout=0, l2=0):
    """Create a pretrained classification model."""

    if nom == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif nom == "ResNet50":
        base = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    else:
        raise ValueError("Modele inconnu")

    base.trainable = False

    entree = tf.keras.Input(shape=(224, 224, 3))
    x = base(entree, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    sortie = tf.keras.layers.Dense(
        CLASSES,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2)
    )(x)

    return tf.keras.Model(entree, sortie), base


def compiler(modele, learning_rate=0.001):
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def callbacks(nom):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(DOSSIER / f"{nom}.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True
        )
    ]


def evaluer(modele, val_ds):
    """Calculate validation Accuracy and Macro F1."""

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = modele(images, training=False)

        y_true.extend(labels.numpy())
        y_pred.extend(
            tf.argmax(predictions, axis=1).numpy()
        )

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return float(accuracy), float(f1)


def tracer_courbes(historique, nom):
    """Save Accuracy and Loss learning curves."""

    for metrique in ["accuracy", "loss"]:
        plt.figure(figsize=(8, 5))

        plt.plot(
            historique.history[metrique],
            label="Train"
        )
        plt.plot(
            historique.history[f"val_{metrique}"],
            label="Validation"
        )

        plt.title(f"{nom} - {metrique}")
        plt.xlabel("Epoch")
        plt.ylabel(metrique)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(RAPPORTS / f"{nom}_{metrique}.png")
        plt.close()


def entrainer(nom, configuration, dropout, l2):
    """Train frozen layers, then perform fine-tuning."""

    print(f"\n===== {nom} - {configuration} =====")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    train_ds = charger_dataset("train", nom)
    val_ds = charger_dataset("val", nom)

    modele, base = creer_modele(nom, dropout, l2)
    resultats = []

    # Phase 1: frozen backbone
    compiler(modele, 0.001)

    nom_phase = f"{nom}_{configuration}_frozen"

    historique = modele.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks(nom_phase)
    )

    modele.load_weights(str(DOSSIER / f"{nom_phase}.weights.h5"))
    tracer_courbes(historique, nom_phase)

    acc, f1 = evaluer(modele, val_ds)
    modele.save(str(DOSSIER / f"{nom_phase}.keras"))

    resultats.append({
        "modele": nom,
        "configuration": configuration,
        "phase": "Frozen",
        "accuracy": acc,
        "f1_macro": f1
    })

    # Phase 2: fine-tuning
    base.trainable = True

    for couche in base.layers:
        couche.trainable = False

    for couche in base.layers[-30:]:
        if not isinstance(couche, tf.keras.layers.BatchNormalization):
            couche.trainable = True

    compiler(modele, 0.00001)

    nom_phase = f"{nom}_{configuration}_finetuned"

    historique = modele.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks(nom_phase)
    )

    modele.load_weights(str(DOSSIER / f"{nom_phase}.weights.h5"))
    tracer_courbes(historique, nom_phase)

    acc, f1 = evaluer(modele, val_ds)
    modele.save(str(DOSSIER / f"{nom_phase}.keras"))

    resultats.append({
        "modele": nom,
        "configuration": configuration,
        "phase": "Fine-tuned",
        "accuracy": acc,
        "f1_macro": f1
    })

    return resultats


def main():
    resultats = []

    # Compare two configurations of MobileNetV2
    configurations = [
        ("baseline", 0.0, 0.0),
        ("regularized", 0.3, 0.0001)
    ]

    for nom, dropout, l2 in configurations:
        resultats.extend(
            entrainer("MobileNetV2", nom, dropout, l2)
        )

    df = pd.DataFrame(resultats)
    df = df.sort_values("f1_macro", ascending=False)

    print("\nComparaison des configurations :")
    print(df.to_string(index=False))

    df.to_csv(RAPPORTS / "comparaison_step4.csv", index=False)

    print("\nMeilleure configuration :")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
