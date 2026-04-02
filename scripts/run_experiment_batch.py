import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
BATCH_SUMMARY_JSON = RESULTS_DIR / "batch_summary.json"


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_experiment(config_path: str) -> None:
    print(f"\n=== Running experiment: {config_path} ===")
    run_command(["python", "scripts/run_experiment.py", config_path])


def compare_experiments() -> None:
    print("\n=== Comparing experiments ===")
    run_command(["python", "scripts/compare_experiments.py"])


def log_to_mlflow(config_path: str) -> None:
    print(f"\n=== Logging to MLflow: {config_path} ===")
    run_command(["python", "scripts/mlflow_run.py", config_path])


def collect_cfgs_from_directory(directory: Path) -> list[str]:
    configs = sorted(str(path.relative_to(ROOT)) for path in directory.glob("*.cfg"))
    if not configs:
        raise ValueError(f"No .cfg files found in directory: {directory}")
    return configs


def normalize_config_paths(args: list[str]) -> list[str]:
    configs: list[str] = []

    for arg in args:
        path = Path(arg)
        abs_path = path if path.is_absolute() else (ROOT / path)

        if abs_path.is_dir():
            configs.extend(collect_cfgs_from_directory(abs_path))
        elif abs_path.is_file() and abs_path.suffix == ".cfg":
            configs.append(str(abs_path.relative_to(ROOT)))
        else:
            raise ValueError(f"Expected a .cfg file or directory, got: {arg}")

    # remove duplicates while preserving order
    seen = set()
    unique_configs: list[str] = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique_configs.append(cfg)

    return unique_configs


def read_best_experiment() -> dict | None:
    if not SUMMARY_CSV.exists():
        return None

    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            return row

    return None


def write_batch_summary(
    configs: list[str],
    successes: list[str],
    failures: list[tuple[str, str]],
    enable_mlflow: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mlflow_enabled": enable_mlflow,
        "requested_configs": configs,
        "successful_runs": successes,
        "failed_runs": [
            {"config": config, "error": error}
            for config, error in failures
        ],
        "best_experiment": read_best_experiment(),
    }

    BATCH_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBatch summary written to: {BATCH_SUMMARY_JSON.relative_to(ROOT)}")


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/run_experiment_batch.py "
            "<config1.cfg|config_dir> <config2.cfg|config_dir> ... [--mlflow]"
        )
        return 1

    raw_args = sys.argv[1:]
    enable_mlflow = False

    if "--mlflow" in raw_args:
        enable_mlflow = True
        raw_args = [arg for arg in raw_args if arg != "--mlflow"]

    configs = normalize_config_paths(raw_args)

    if not configs:
        print("No config files provided.")
        return 1

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for config_path in configs:
        try:
            run_experiment(config_path)
            successes.append(config_path)

            if enable_mlflow:
                log_to_mlflow(config_path)

        except Exception as exc:
            failures.append((config_path, str(exc)))
            print(f"\nFAILED: {config_path}")
            print(exc)

    if successes:
        compare_experiments()

    write_batch_summary(configs, successes, failures, enable_mlflow)

    print("\n=== Batch Summary ===")
    print(f"Successful runs: {len(successes)}")
    for config in successes:
        print(f"  OK  - {config}")

    print(f"Failed runs: {len(failures)}")
    for config, error in failures:
        print(f"  ERR - {config}")
        print(f"        {error}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())