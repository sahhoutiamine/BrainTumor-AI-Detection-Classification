
import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.models.classifier import (
    unfreeze_last_layers,
    compile_classifier,
    CLASS_NAMES,
)
from src.training.data_loader import (
    create_model_datasets,
)
from src.training.train_classifier import (
    get_validation_metrics,
)


MODEL_DIR = Path("models/classification")
REPORT_DIR = Path("reports/metrics")


def finetune_classifier(
    model_name="mobilenetv2",
    epochs=5,
    batch_size=32,
    unfreeze_layers=30,
):
    """
    Fine-tune the last selected
    pretrained backbone layers.
    """

    tf.keras.utils.set_random_seed(42)

    frozen_path = (
        MODEL_DIR
        / f"{model_name}_frozen.keras"
    )

    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Train the frozen model first: "
            f"{frozen_path}"
        )

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

    model = tf.keras.models.load_model(
        str(frozen_path)
    )

    model = unfreeze_last_layers(
        model,
        number_of_layers=unfreeze_layers,
    )

    # Recompile after changing
    # the trainable layers.

    model = compile_classifier(
        model,
        learning_rate=0.00001,
    )

    model.summary()

    model_path = (
        MODEL_DIR
        / f"{model_name}_finetuned.keras"
    )

    csv_path = (
        REPORT_DIR
        / f"{model_name}_finetuned_history.csv"
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
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(csv_path)
        ),
    ]

    print("\nFine-tuning classifier")
    print("-" * 40)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

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
        "phase": "finetuned",
        "epochs_completed": len(
            history.history["loss"]
        ),
        "unfreeze_layers": unfreeze_layers,
        "learning_rate": 0.00001,
        "selection_metric": "val_loss",
        "class_names": CLASS_NAMES,
        "model_path": str(model_path),
        **validation_metrics,
    }

    report_path = (
        REPORT_DIR
        / f"{model_name}_finetuned_metrics.json"
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

    print("\nFine-tuning validation results")

    for key, value in validation_metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    print(
        f"\nModel saved: {model_path}"
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
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    finetune_classifier(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        unfreeze_layers=args.unfreeze_layers,
    )


if __name__ == "__main__":
    main()
