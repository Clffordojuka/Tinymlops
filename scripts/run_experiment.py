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
ARCHIVE_DIR = ROOT / "metrics" / "archive"
CHECKPOINT_ARCHIVE_DIR = ROOT / "models" / "registry"


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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def archive_outputs(experiment_name: str) -> tuple[Path, Path, Path]:
    ensure_dir(ARCHIVE_DIR)
    ensure_dir(CHECKPOINT_ARCHIVE_DIR)

    train_archive = ARCHIVE_DIR / f"{experiment_name}_train_metrics.json"
    eval_archive = ARCHIVE_DIR / f"{experiment_name}_eval_metrics.json"
    checkpoint_archive = CHECKPOINT_ARCHIVE_DIR / f"{experiment_name}_model.txt"

    shutil.copy2(TRAIN_METRICS_PATH, train_archive)
    shutil.copy2(EVAL_METRICS_PATH, eval_archive)
    shutil.copy2(CHECKPOINT_PATH, checkpoint_archive)

    return train_archive, eval_archive, checkpoint_archive


def experiment_name_from_config(config_path: str) -> str:
    return Path(config_path).stem


def print_summary(config_path: str, train_metrics: dict, eval_metrics: dict) -> None:
    print("\n=== Experiment Summary ===")
    print(f"Config: {config_path}")
    print(f"Train final_loss: {train_metrics.get('final_loss')}")
    print(f"Train epochs: {train_metrics.get('epochs')}")
    print(f"Train learning_rate: {train_metrics.get('learning_rate')}")
    print(f"Eval eval_loss: {eval_metrics.get('eval_loss')}")
    print(f"Prediction x=4: {eval_metrics.get('prediction_x4')}")
    print(f"Weight: {eval_metrics.get('weight')}")
    print(f"Bias: {eval_metrics.get('bias')}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <config_path>")
        return 1

    config_path = sys.argv[1]
    experiment_name = experiment_name_from_config(config_path)

    write_params(config_path)
    run_dvc_repro()

    train_metrics = load_json(TRAIN_METRICS_PATH)
    eval_metrics = load_json(EVAL_METRICS_PATH)

    train_archive, eval_archive, checkpoint_archive = archive_outputs(experiment_name)

    print_summary(config_path, train_metrics, eval_metrics)

    print("\nArchived outputs:")
    print(f"  Train metrics: {train_archive.relative_to(ROOT)}")
    print(f"  Eval metrics: {eval_archive.relative_to(ROOT)}")
    print(f"  Checkpoint: {checkpoint_archive.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())