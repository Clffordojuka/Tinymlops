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

    architecture = model_params.get("architecture", {})
    parameters = model_params.get("parameters", {})

    mlflow.set_experiment("tinyml-ops")

    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("experiment_name", experiment_name)

        mlflow.log_param("model_type", train_metrics.get("model_type"))
        mlflow.log_param("hidden_dim", train_metrics.get("hidden_dim"))
        mlflow.log_param("hidden_layers", train_metrics.get("hidden_layers"))
        mlflow.log_param("hidden_activation", train_metrics.get("hidden_activation"))
        mlflow.log_param("num_layers", train_metrics.get("num_layers"))

        mlflow.log_param("epochs", train_metrics.get("epochs"))
        mlflow.log_param("learning_rate", train_metrics.get("learning_rate"))
        mlflow.log_param("final_learning_rate", train_metrics.get("final_learning_rate"))

        mlflow.log_param("optimizer", train_metrics.get("optimizer"))
        mlflow.log_param("adam_beta1", train_metrics.get("adam_beta1"))
        mlflow.log_param("adam_beta2", train_metrics.get("adam_beta2"))
        mlflow.log_param("adam_epsilon", train_metrics.get("adam_epsilon"))

        mlflow.log_param("lr_schedule", train_metrics.get("lr_schedule"))
        mlflow.log_param("lr_step_size", train_metrics.get("lr_step_size"))
        mlflow.log_param("lr_decay", train_metrics.get("lr_decay"))
        mlflow.log_param("l2_lambda", train_metrics.get("l2_lambda"))
        mlflow.log_param("weight_init", train_metrics.get("weight_init"))

        mlflow.log_param("batch_size", train_metrics.get("batch_size"))
        mlflow.log_param("validation_split", train_metrics.get("validation_split"))
        mlflow.log_param("shuffle", train_metrics.get("shuffle"))
        mlflow.log_param("split_seed", train_metrics.get("split_seed"))

        mlflow.log_param("input_dim", architecture.get("input_dim"))
        mlflow.log_param("output_dim", architecture.get("output_dim"))
        mlflow.log_param("architecture_hidden_dim", architecture.get("hidden_dim"))
        mlflow.log_param("architecture_hidden_layers", architecture.get("hidden_layers"))
        mlflow.log_param("architecture_hidden_activation", architecture.get("hidden_activation"))
        mlflow.log_param("architecture_num_layers", architecture.get("num_layers"))

        mlflow.log_metric("train_loss", train_metrics.get("train_loss"))
        mlflow.log_metric("val_loss", train_metrics.get("val_loss"))
        mlflow.log_metric("eval_loss", eval_metrics.get("eval_loss"))
        mlflow.log_metric("prediction_x4", eval_metrics.get("prediction_x4"))
        mlflow.log_metric("parameter_count", eval_metrics.get("parameter_count"))
        mlflow.log_metric("weight_l2_norm", eval_metrics.get("weight_l2_norm"))
        mlflow.log_metric("max_abs_weight", eval_metrics.get("max_abs_weight"))
        mlflow.log_metric("bias_l2_norm", eval_metrics.get("bias_l2_norm"))

        if train_metrics.get("best_val_loss") is not None:
            mlflow.log_metric("best_val_loss", train_metrics.get("best_val_loss"))
        if train_metrics.get("best_epoch") is not None:
            mlflow.log_metric("best_epoch", train_metrics.get("best_epoch"))

        mlflow.set_tag("model_type", train_metrics.get("model_type", "unknown"))
        mlflow.set_tag("architecture", model_params.get("model_type", "unknown"))
        mlflow.set_tag("optimizer", str(train_metrics.get("optimizer", "unknown")))
        mlflow.set_tag("weight_init", str(train_metrics.get("weight_init", "unknown")))

        model_type = model_params.get("model_type")

        if model_type == "mlp":
            mlflow.set_tag("has_hidden_layer", "true")
            mlflow.set_tag("has_multiple_hidden_layers", "false")
            mlflow.log_param("exported_hidden_dim", architecture.get("hidden_dim"))
            mlflow.log_param("hidden_weight_rows", len(parameters.get("hidden_weights", [])))
            mlflow.log_param(
                "hidden_weight_cols",
                len(parameters.get("hidden_weights", [[]])[0]) if parameters.get("hidden_weights") else 0
            )
            mlflow.log_param("output_weight_rows", len(parameters.get("output_weights", [])))
            mlflow.log_param(
                "output_weight_cols",
                len(parameters.get("output_weights", [[]])[0]) if parameters.get("output_weights") else 0
            )
        elif model_type == "deep_mlp":
            mlflow.set_tag("has_hidden_layer", "true")
            mlflow.set_tag("has_multiple_hidden_layers", "true")
            layers = parameters.get("layers", [])
            mlflow.log_param("exported_num_layers", len(layers))
            for i, layer in enumerate(layers):
                mlflow.log_param(f"layer_{i}_input_dim", layer.get("input_dim"))
                mlflow.log_param(f"layer_{i}_output_dim", layer.get("output_dim"))
        else:
            mlflow.set_tag("has_hidden_layer", "false")
            mlflow.set_tag("has_multiple_hidden_layers", "false")
            mlflow.log_param("weight_rows", len(parameters.get("weights", [])))
            mlflow.log_param(
                "weight_cols",
                len(parameters.get("weights", [[]])[0]) if parameters.get("weights") else 0
            )

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