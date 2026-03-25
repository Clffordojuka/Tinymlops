import json
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "metrics" / "archive"
CHECKPOINT_ARCHIVE_DIR = ROOT / "models" / "registry"
MODEL_EXPORT_ARCHIVE_DIR = ROOT / "models" / "exported"

MLFLOW_DB_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_DB_URI)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_name_from_config(config_path: str) -> str:
    return Path(config_path).stem


def run_experiment_script(config_path: str) -> None:
    cmd = ["python", "scripts/run_experiment.py", config_path]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("run_experiment.py failed")


def find_archived_outputs(experiment_name: str) -> tuple[Path, Path, Path, Path]:
    train_metrics = ARCHIVE_DIR / f"{experiment_name}_train_metrics.json"
    eval_metrics = ARCHIVE_DIR / f"{experiment_name}_eval_metrics.json"
    checkpoint = CHECKPOINT_ARCHIVE_DIR / f"{experiment_name}_model.txt"
    model_params = MODEL_EXPORT_ARCHIVE_DIR / f"{experiment_name}_model_params.json"

    for path in (train_metrics, eval_metrics, checkpoint, model_params):
        if not path.exists():
            raise FileNotFoundError(f"Missing archived output: {path}")

    return train_metrics, eval_metrics, checkpoint, model_params


def log_run(config_path: str) -> None:
    experiment_name = experiment_name_from_config(config_path)

    run_experiment_script(config_path)

    train_metrics_path, eval_metrics_path, checkpoint_path, model_params_path = find_archived_outputs(experiment_name)

    train_metrics = load_json(train_metrics_path)
    eval_metrics = load_json(eval_metrics_path)
    model_params = load_json(model_params_path)

    mlflow.set_experiment("tinyml-ops")

    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("epochs", train_metrics.get("epochs"))
        mlflow.log_param("learning_rate", train_metrics.get("learning_rate"))
        mlflow.log_param("lr_schedule", train_metrics.get("lr_schedule"))
        mlflow.log_param("lr_step_size", train_metrics.get("lr_step_size"))
        mlflow.log_param("lr_decay", train_metrics.get("lr_decay"))
        mlflow.log_param("batch_size", train_metrics.get("batch_size"))
        mlflow.log_param("validation_split", train_metrics.get("validation_split"))
        mlflow.log_param("shuffle", train_metrics.get("shuffle"))
        mlflow.log_param("split_seed", train_metrics.get("split_seed"))
        mlflow.log_param("input_dim", model_params.get("input_dim"))
        mlflow.log_param("output_dim", model_params.get("output_dim"))

        mlflow.log_metric("train_loss", train_metrics.get("train_loss"))
        mlflow.log_metric("val_loss", train_metrics.get("val_loss"))
        mlflow.log_metric("eval_loss", eval_metrics.get("eval_loss"))
        mlflow.log_metric("prediction_x4", eval_metrics.get("prediction_x4"))
        mlflow.log_metric("parameter_count", eval_metrics.get("parameter_count"))
        mlflow.log_metric("weight_l2_norm", eval_metrics.get("weight_l2_norm"))
        mlflow.log_metric("max_abs_weight", eval_metrics.get("max_abs_weight"))
        mlflow.log_metric("bias_l2_norm", eval_metrics.get("bias_l2_norm"))
        mlflow.log_metric("final_learning_rate", train_metrics.get("final_learning_rate"))

        mlflow.log_artifact(str(train_metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(eval_metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(model_params_path), artifact_path="model_params")

        summary_csv = ROOT / "results" / "experiment_summary.csv"
        if summary_csv.exists():
            mlflow.log_artifact(str(summary_csv), artifact_path="reports")

    print(f"Logged MLflow run for: {experiment_name}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/mlflow_run.py <config_path>")
        return 1

    config_path = sys.argv[1]
    log_run(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())