
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.classifier import (
    build_classifier,
    compile_classifier,
    CLASS_NAMES,
)
from src.training.data_loader import (
    create_model_datasets,
)


MODEL_DIR = Path("models/classification")
REPORT_DIR = Path("reports/metrics")


def get_validation_metrics(model, val_data):
    """
    Evaluate on validation only.
    """

    y_true = np.asarray(
        val_data.labels,
        dtype=np.int32,
    )

    probabilities = model.predict(
        val_data,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "val_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "val_precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "val_recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "val_f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_classifier(
    model_name="mobilenetv2",
    epochs=10,
    batch_size=32,
):
    """
    Train the new classification head
    with frozen pretrained layers.
    """

    tf.keras.utils.set_random_seed(42)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data, val_data, _ = (
        create_model_datasets(
            model_name,
            batch_size=batch_size,
        )
    )

    model = build_classifier(
        model_name
    )

    model = compile_classifier(
        model,
        learning_rate=0.001,
    )

    model.summary()

    model_path = (
        MODEL_DIR
        / f"{model_name}_frozen.keras"
    )

    csv_path = (
        REPORT_DIR
        / f"{model_name}_frozen_history.csv"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(csv_path)
        ),
    ]

    print("\nTraining classifier")
    print("-" * 40)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload the checkpoint with the
    # lowest validation loss.

    best_model = tf.keras.models.load_model(
        str(model_path)
    )

    validation_metrics = (
        get_validation_metrics(
            best_model,
            val_data,
        )
    )

    report = {
        "model": model_name,
        "phase": "frozen",
        "epochs_completed": len(
            history.history["loss"]
        ),
        "selection_metric": "val_loss",
        "class_names": CLASS_NAMES,
        "model_path": str(model_path),
        **validation_metrics,
    }

    report_path = (
        REPORT_DIR
        / f"{model_name}_frozen_metrics.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nValidation results")

    for key, value in validation_metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    print(
        f"\nBest model saved: {model_path}"
    )

    print(
        f"Metrics saved: {report_path}"
    )

    return best_model, history


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "mobilenetv2",
            "resnet50",
        ],
        default="mobilenetv2",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    train_classifier(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
