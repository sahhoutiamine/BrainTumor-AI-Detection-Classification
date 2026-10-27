
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("reports/metrics")
FIGURE_DIR = Path("reports/figures")

EXPERIMENTS = [
    ("mobilenetv2", "frozen"),
    ("mobilenetv2", "finetuned"),
    ("resnet50", "frozen"),
    ("resnet50", "finetuned"),
]


def load_experiment_results():
    """
    Read the saved validation metrics.
    """

    results = []

    for model_name, phase in EXPERIMENTS:

        report_path = (
            REPORT_DIR
            / f"{model_name}_{phase}_metrics.json"
        )

        if not report_path.exists():

            print(
                f"Missing report: {report_path}"
            )

            continue

        with open(
            report_path,
            "r",
            encoding="utf-8",
        ) as file:

            report = json.load(file)

        results.append({
            "experiment": (
                f"{model_name}_{phase}"
            ),
            "architecture": model_name,
            "phase": phase,
            "accuracy": report[
                "val_accuracy"
            ],
            "precision": report[
                "val_precision_macro"
            ],
            "recall": report[
                "val_recall_macro"
            ],
            "f1_score": report[
                "val_f1_macro"
            ],
        })

    return pd.DataFrame(results)


def plot_comparison(results):
    """
    Compare validation metrics
    using bar charts.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    ax = results.set_index(
        "experiment"
    )[metrics].plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Transfer Learning Model Comparison"
    )

    ax.set_xlabel(
        "Model configuration"
    )

    ax.set_ylabel(
        "Validation score"
    )

    ax.set_ylim(0, 1.05)

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "classifier_comparison.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def plot_training_curves():
    """
    Plot training and validation accuracy
    from the saved CSV histories.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, phase in EXPERIMENTS:

        csv_path = (
            REPORT_DIR
            / f"{model_name}_{phase}_history.csv"
        )

        if not csv_path.exists():
            continue

        history = pd.read_csv(
            csv_path
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            history["epoch"] + 1,
            history["accuracy"],
            label="Training accuracy",
        )

        plt.plot(
            history["epoch"] + 1,
            history["val_accuracy"],
            label="Validation accuracy",
        )

        plt.title(
            f"{model_name} - {phase}"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.ylim(0, 1.05)

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"{model_name}_{phase}_accuracy.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()


def main():

    results = load_experiment_results()

    if results.empty:

        print(
            "No experiment reports found."
        )

        return

    # Higher macro F1 is better.
    # Accuracy breaks ties.

    results = results.sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )

    print("\nMODEL COMPARISON")
    print("-" * 50)

    print(
        results.to_string(index=False)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        REPORT_DIR
        / "classifier_comparison.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    best = results.iloc[0]

    recommendation = {
        "experiment": best["experiment"],
        "architecture": best["architecture"],
        "phase": best["phase"],
        "selection_metric": "val_f1_macro",
        "validation_f1_macro": float(
            best["f1_score"]
        ),
        "validation_accuracy": float(
            best["accuracy"]
        ),
    }

    json_path = (
        REPORT_DIR
        / "selected_classifier.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            recommendation,
            file,
            indent=4,
        )

    plot_comparison(results)

    plot_training_curves()

    print("\nBEST VALIDATION CONFIGURATION")
    print("-" * 50)

    print(
        "Model:",
        best["experiment"],
    )

    print(
        "Validation Macro F1:",
        round(best["f1_score"], 4),
    )

    print(
        "Validation Accuracy:",
        round(best["accuracy"], 4),
    )

    print(
        "\nComparison saved:",
        csv_path,
    )


if __name__ == "__main__":
    main()
