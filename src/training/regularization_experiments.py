
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data.preprocessing import (
    MRIDataGenerator,
    get_image_paths,
    preprocess_image,
)

from src.models.classifier import (
    build_classifier,
    compile_classifier,
    CLASS_NAMES,
)


MODEL_DIR = Path("models/classification")
METRICS_DIR = Path("reports/metrics")

SEED = 42

CONFIGURATIONS = {
    "baseline": {
        "augmentation": False,
        "dropout": 0.0,
        "l2": 0.0,
    },
    "regularized": {
        "augmentation": True,
        "dropout": 0.4,
        "l2": 0.001,
    },
}


# -------------------------------------
# Custom image generator
# -------------------------------------

class ExperimentDataGenerator(MRIDataGenerator):
    """
    Load images using the same Step 2
    preprocessing pipeline.

    Allow augmentation to be enabled
    or disabled independently.
    """

    def __init__(
        self,
        image_paths,
        labels,
        batch_size=32,
        training=False,
        augmentation=False,
        **kwargs,
    ):

        super().__init__(
            image_paths=image_paths,
            labels=labels,
            batch_size=batch_size,
            training=training,
            **kwargs,
        )

        self.augmentation = augmentation

    def __getitem__(self, index):

        start = index * self.batch_size
        end = start + self.batch_size

        batch_indices = self.indices[start:end]

        images = []
        labels = []

        for image_index in batch_indices:

            image_path = self.image_paths[
                image_index
            ]

            label = self.labels[
                image_index
            ]

            image = preprocess_image(
                image_path,
                training=(
                    self.training
                    and self.augmentation
                ),
            )

            images.append(image)
            labels.append(label)

        X = np.stack(
            images
        ).astype(np.float32)

        y = np.array(
            labels,
            dtype=np.int32,
        )

        return X, y


# -------------------------------------
# Create datasets
# -------------------------------------

def create_experiment_datasets(
    batch_size,
    augmentation,
):
    """
    Only training can use augmentation.
    Validation is always deterministic.
    """

    train_paths, train_labels = get_image_paths(
        "train"
    )

    val_paths, val_labels = get_image_paths(
        "val"
    )

    train_data = ExperimentDataGenerator(
        train_paths,
        train_labels,
        batch_size=batch_size,
        training=True,
        augmentation=augmentation,
        seed=SEED,
    )

    val_data = ExperimentDataGenerator(
        val_paths,
        val_labels,
        batch_size=batch_size,
        training=False,
        augmentation=False,
        seed=SEED,
    )

    return train_data, val_data


# -------------------------------------
# Training callbacks
# -------------------------------------

def create_callbacks(
    model_path,
    history_path,
):
    """
    Configure EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau and CSVLogger.
    """

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
            mode="min",
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_delta=0.001,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            filename=str(history_path)
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks


# -------------------------------------
# Measure validation performance
# -------------------------------------

def evaluate_validation(model, val_data):
    """
    Compute predictive metrics on
    the validation dataset only.
    """

    probabilities = model.predict(
        val_data,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    actual_labels = np.array(
        val_data.labels,
        dtype=np.int32,
    )

    results = {
        "accuracy": float(
            accuracy_score(
                actual_labels,
                predictions,
            )
        ),

        "precision_macro": float(
            precision_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return results


# -------------------------------------
# Run one experiment
# -------------------------------------

def run_experiment(
    name,
    epochs=20,
    batch_size=32,
):
    """
    Train one configuration and save
    its checkpoint, history and metrics.
    """

    if name not in CONFIGURATIONS:
        raise ValueError(
            f"Unknown configuration: {name}"
        )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = CONFIGURATIONS[name]

    print("\n" + "=" * 50)
    print("EXPERIMENT:", name.upper())
    print("=" * 50)

    print("Augmentation:", config["augmentation"])
    print("Dropout:", config["dropout"])
    print("L2:", config["l2"])

    train_data, val_data = (
        create_experiment_datasets(
            batch_size=batch_size,
            augmentation=config["augmentation"],
        )
    )

    model = build_classifier(
        model_name="mobilenetv2",
        dropout_rate=config["dropout"],
        l2_value=config["l2"],
    )

    model = compile_classifier(
        model,
        learning_rate=0.001,
    )

    model_path = (
        MODEL_DIR
        / f"mobilenetv2_{name}_step4.keras"
    )

    history_path = (
        METRICS_DIR
        / f"mobilenetv2_{name}_step4_history.csv"
    )

    metrics_path = (
        METRICS_DIR
        / f"mobilenetv2_{name}_step4_metrics.json"
    )

    callbacks = create_callbacks(
        model_path,
        history_path,
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the checkpoint corresponding
    # to the lowest validation loss.
    best_model = tf.keras.models.load_model(
        str(model_path)
    )

    metrics = evaluate_validation(
        best_model,
        val_data,
    )

    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        )
    ) + 1

    report = {
        "experiment": name,
        "architecture": "mobilenetv2",
        "augmentation": config["augmentation"],
        "dropout": config["dropout"],
        "l2": config["l2"],
        "learning_rate": 0.001,
        "epochs_completed": len(
            history.history["loss"]
        ),
        "best_epoch": best_epoch,
        "best_val_loss": float(
            min(history.history["val_loss"])
        ),
        "class_names": CLASS_NAMES,
        "model_path": str(model_path),
        **metrics,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nValidation metrics")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("Best epoch:", best_epoch)
    print("Saved model:", model_path)

    return report


# -------------------------------------
# Compare both experiments
# -------------------------------------

def compare_experiments():
    """
    Load metrics and create a
    comparison CSV.
    """

    reports = []

    for name in CONFIGURATIONS:

        path = (
            METRICS_DIR
            / f"mobilenetv2_{name}_step4_metrics.json"
        )

        if not path.exists():
            print("Missing experiment:", name)
            continue

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            reports.append(
                json.load(file)
            )

    if not reports:
        print("No results available.")
        return

    dataframe = pd.DataFrame(
        reports
    )

    columns = [
        "experiment",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "best_epoch",
    ]

    print("\nMODEL COMPARISON")
    print("-" * 50)

    print(
        dataframe[columns].to_string(
            index=False
        )
    )

    output_path = (
        METRICS_DIR
        / "step4_comparison.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    if len(reports) == len(CONFIGURATIONS):

        best = max(
            reports,
            key=lambda row: (
                row["f1_macro"],
                row["accuracy"],
            ),
        )

        print(
            "\nBest validation configuration:",
            best["experiment"],
        )

        print(
            "Validation Macro F1:",
            round(best["f1_macro"], 4),
        )

        print(
            "Model:",
            best["model_path"],
        )

    else:
        print(
            "\nComparison incomplete: "
            "run both experiments."
        )


# -------------------------------------
# Main
# -------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "baseline",
            "regularized",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.experiment == "all":

        run_experiment(
            "baseline",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        run_experiment(
            "regularized",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    else:

        run_experiment(
            args.experiment,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    compare_experiments()


if __name__ == "__main__":
    main()
