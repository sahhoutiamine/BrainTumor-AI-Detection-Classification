
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("reports/metrics")
FIGURE_DIR = Path("reports/figures")

EXPERIMENTS = [
    "baseline",
    "regularized",
]


def load_history(experiment):
    """
    Load the CSV history saved by Keras.
    """

    path = (
        METRICS_DIR
        / f"mobilenetv2_{experiment}_step4_history.csv"
    )

    if not path.exists():
        print("Missing history:", path)
        return None

    return pd.read_csv(path)


def plot_learning_curves(experiment):
    """
    Create Accuracy and Loss graphs.
    """

    history = load_history(experiment)

    if history is None:
        return

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"] + 1

    # Accuracy curve
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        f"{experiment.capitalize()} - Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    accuracy_path = (
        FIGURE_DIR
        / f"step4_{experiment}_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
    )

    plt.close()

    # Loss curve
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        f"{experiment.capitalize()} - Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    loss_path = (
        FIGURE_DIR
        / f"step4_{experiment}_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
    )

    plt.close()

    print("Saved:", accuracy_path)
    print("Saved:", loss_path)


def analyze_history(experiment):
    """
    Print useful indicators for detecting
    possible overfitting or underfitting.
    """

    history = load_history(experiment)

    if history is None:
        return

    last_train_accuracy = float(
        history["accuracy"].iloc[-1]
    )

    last_val_accuracy = float(
        history["val_accuracy"].iloc[-1]
    )

    best_val_accuracy = float(
        history["val_accuracy"].max()
    )

    best_val_loss = float(
        history["val_loss"].min()
    )

    accuracy_gap = (
        last_train_accuracy
        - last_val_accuracy
    )

    print("\nExperiment:", experiment)
    print("-" * 40)

    print(
        "Last training accuracy:",
        round(last_train_accuracy, 4),
    )

    print(
        "Last validation accuracy:",
        round(last_val_accuracy, 4),
    )

    print(
        "Best validation accuracy:",
        round(best_val_accuracy, 4),
    )

    print(
        "Best validation loss:",
        round(best_val_loss, 4),
    )

    print(
        "Final accuracy gap:",
        round(accuracy_gap, 4),
    )

    if (
        accuracy_gap > 0.10
        and last_train_accuracy > 0.80
    ):

        print(
            "Possible overfitting: "
            "training accuracy is much higher "
            "than validation accuracy."
        )

    elif (
        last_train_accuracy < 0.65
        and last_val_accuracy < 0.65
    ):

        print(
            "Possible underfitting: "
            "both accuracies are relatively low."
        )

    else:

        print(
            "No strong warning from these "
            "simple accuracy thresholds."
        )

    print(
        "Inspect the Loss curves before "
        "making a final conclusion."
    )


def plot_experiment_comparison():
    """
    Compare the predictive metrics
    of both configurations.
    """

    path = (
        METRICS_DIR / "step4_comparison.csv"
    )

    if not path.exists():
        print("Missing comparison:", path)
        return

    dataframe = pd.read_csv(path)

    if len(dataframe) < 2:
        print("Run both experiments first.")
        return

    columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    ax = dataframe.set_index(
        "experiment"
    )[columns].plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Baseline vs Regularized MobileNetV2"
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Validation Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / "step4_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print("Saved:", output_path)


def main():

    for experiment in EXPERIMENTS:

        plot_learning_curves(experiment)

        analyze_history(experiment)

    plot_experiment_comparison()

    print("\nStep 4 analysis completed.")


if __name__ == "__main__":
    main()
