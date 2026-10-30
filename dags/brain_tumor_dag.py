
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from airflow.sdk import dag, task


PROJECT = Path(__file__).resolve().parents[1]


@dag(
    dag_id="brain_tumor_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2)
    }
)
def brain_tumor_pipeline():

    @task
    def executer(module):
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT,
            check=True
        )

    exploration = executer.override(task_id="exploration")(
        "src.data.exploration"
    )

    preparation = executer.override(task_id="preparation")(
        "src.data.preparation"
    )

    preprocessing = executer.override(task_id="preprocessing")(
        "src.data.preprocessing"
    )

    classification = executer.override(task_id="classification")(
        "src.training.train"
    )

    detection = executer.override(task_id="detection")(
        "src.training.train_yolo"
    )

    evaluation_classification = executer.override(
        task_id="evaluation_classification"
    )(
        "src.evaluation.evaluate_classifier"
    )

    evaluation_detection = executer.override(
        task_id="evaluation_detection"
    )(
        "src.evaluation.evaluate_detector"
    )

    exploration >> preparation >> preprocessing
    preprocessing >> classification >> detection
    detection >> evaluation_classification >> evaluation_detection


brain_tumor_pipeline()
