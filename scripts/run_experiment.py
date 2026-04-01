import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
TRAIN_METRICS_PATH = ROOT / "metrics" / "train_metrics.json"
EVAL_METRICS_PATH = ROOT / "metrics" / "eval_metrics.json"
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "linear_model.txt"
MODEL_PARAMS_PATH = ROOT / "models" / "exported" / "model_params.json"
ARCHIVE_DIR = ROOT / "metrics" / "archive"
CHECKPOINT_ARCHIVE_DIR = ROOT / "models" / "registry"
MODEL_EXPORT_ARCHIVE_DIR = ROOT / "models" / "exported"


def write_params(config_path: str) -> None:
    content = (
        f"train:\n"
        f"  config_path: {config_path}\n\n"
        f"evaluate:\n"
        f"  config_path: {config_path}\n"
    )
    PARAMS_PATH.write_text(content, encoding="utf-8")


def run_dvc_repro() -> None:
    cmd = ["python", "-m", "dvc", "repro"]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("DVC repro failed.")


def export_model_params() -> None:
    cmd = [
        "python",
        "scripts/export_model_params.py",
        str(CHECKPOINT_PATH),
        str(MODEL_PARAMS_PATH),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("Model parameter export failed.")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def archive_outputs(experiment_name: str) -> tuple[Path, Path, Path, Path]:
    ensure_dir(ARCHIVE_DIR)
    ensure_dir(CHECKPOINT_ARCHIVE_DIR)
    ensure_dir(MODEL_EXPORT_ARCHIVE_DIR)

    train_archive = ARCHIVE_DIR / f"{experiment_name}_train_metrics.json"
    eval_archive = ARCHIVE_DIR / f"{experiment_name}_eval_metrics.json"
    checkpoint_archive = CHECKPOINT_ARCHIVE_DIR / f"{experiment_name}_model.txt"
    model_params_archive = MODEL_EXPORT_ARCHIVE_DIR / f"{experiment_name}_model_params.json"

    shutil.copy2(TRAIN_METRICS_PATH, train_archive)
    shutil.copy2(EVAL_METRICS_PATH, eval_archive)
    shutil.copy2(CHECKPOINT_PATH, checkpoint_archive)
    shutil.copy2(MODEL_PARAMS_PATH, model_params_archive)

    return train_archive, eval_archive, checkpoint_archive, model_params_archive


def experiment_name_from_config(config_path: str) -> str:
    return Path(config_path).stem


def print_model_params(model_params: dict) -> None:
    model_type = model_params.get("model_type")
    architecture = model_params.get("architecture", {})
    parameters = model_params.get("parameters", {})

    print(f"Model type: {model_type}")
    print(f"Model input_dim: {architecture.get('input_dim')}")
    print(f"Model hidden_dim: {architecture.get('hidden_dim')}")
    print(f"Model hidden_activation: {architecture.get('hidden_activation')}")
    print(f"Model output_dim: {architecture.get('output_dim')}")

    if model_type == "mlp":
        print(f"Hidden weights: {parameters.get('hidden_weights')}")
        print(f"Hidden bias: {parameters.get('hidden_bias')}")
        print(f"Output weights: {parameters.get('output_weights')}")
        print(f"Output bias: {parameters.get('output_bias')}")
    else:
        print(f"Weights: {parameters.get('weights')}")
        print(f"Bias: {parameters.get('bias')}")


def print_summary(config_path: str, train_metrics: dict, eval_metrics: dict, model_params: dict) -> None:
    print("\n=== Experiment Summary ===")
    print(f"Config: {config_path}")
    print(f"Train loss: {train_metrics.get('train_loss')}")
    print(f"Val loss: {train_metrics.get('val_loss')}")
    print(f"Train epochs: {train_metrics.get('epochs')}")
    print(f"Train learning_rate: {train_metrics.get('learning_rate')}")
    print(f"LR schedule: {train_metrics.get('lr_schedule')}")
    print(f"LR step size: {train_metrics.get('lr_step_size')}")
    print(f"LR decay: {train_metrics.get('lr_decay')}")
    print(f"L2 lambda: {train_metrics.get('l2_lambda')}")
    print(f"Model type: {train_metrics.get('model_type')}")
    print(f"Hidden dim: {train_metrics.get('hidden_dim')}")
    print(f"Hidden activation: {train_metrics.get('hidden_activation')}")
    print(f"Final learning_rate: {train_metrics.get('final_learning_rate')}")
    print(f"Batch size: {train_metrics.get('batch_size')}")
    print(f"Validation split: {train_metrics.get('validation_split')}")
    print(f"Shuffle: {train_metrics.get('shuffle')}")
    print(f"Split seed: {train_metrics.get('split_seed')}")
    print(f"Best val loss: {train_metrics.get('best_val_loss')}")
    print(f"Best epoch: {train_metrics.get('best_epoch')}")
    print(f"Stopped early: {train_metrics.get('stopped_early')}")
    print(f"Patience: {train_metrics.get('patience')}")
    print(f"Min delta: {train_metrics.get('min_delta')}")
    print(f"Save best only: {train_metrics.get('save_best_only')}")
    print(f"Eval eval_loss: {eval_metrics.get('eval_loss')}")
    print(f"Prediction x=4: {eval_metrics.get('prediction_x4')}")
    print_model_params(model_params)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <config_path>")
        return 1

    config_path = sys.argv[1]
    experiment_name = experiment_name_from_config(config_path)

    write_params(config_path)
    run_dvc_repro()
    export_model_params()

    train_metrics = load_json(TRAIN_METRICS_PATH)
    eval_metrics = load_json(EVAL_METRICS_PATH)
    model_params = load_json(MODEL_PARAMS_PATH)

    train_archive, eval_archive, checkpoint_archive, model_params_archive = archive_outputs(experiment_name)

    print_summary(config_path, train_metrics, eval_metrics, model_params)

    print("\nArchived outputs:")
    print(f"  Train metrics: {train_archive.relative_to(ROOT)}")
    print(f"  Eval metrics: {eval_archive.relative_to(ROOT)}")
    print(f"  Checkpoint: {checkpoint_archive.relative_to(ROOT)}")
    print(f"  Model params: {model_params_archive.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())