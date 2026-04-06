import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs" / "experiments"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
PROMOTED_CONFIG = CONFIGS_DIR / "best_default.cfg"
PROMOTION_JSON = RESULTS_DIR / "promoted_default.json"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_text(value: str, default: str = "unknown") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def is_legacy_or_malformed_name(name: str) -> bool:
    if not name:
        return True

    stripped = name.strip()
    lowered = stripped.lower()
    if lowered != stripped:
        return True

    suspicious = [
        "unknown",
    ]
    for token in suspicious:
        if token in stripped:
            return True

    return False


def classify_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = []
    excluded = []

    for row in rows:
        eval_loss = parse_float(row.get("eval_loss"))
        train_loss = parse_float(row.get("train_loss"))
        val_loss = parse_float(row.get("val_loss"))

        model_type = normalize_text(row.get("model_type"))
        hidden_activation = normalize_text(row.get("hidden_activation"), "none")
        optimizer = normalize_text(row.get("optimizer"))
        weight_init = normalize_text(row.get("weight_init"))
        experiment = normalize_text(row.get("experiment"), "")

        enriched = {
            **row,
            "_eval_loss": eval_loss,
            "_train_loss": train_loss,
            "_val_loss": val_loss,
            "_model_type": model_type,
            "_hidden_activation": hidden_activation,
            "_optimizer": optimizer,
            "_weight_init": weight_init,
            "_experiment": experiment,
            "_exclude_reason": "",
        }

        reason = None

        if eval_loss is None or eval_loss < 0.0:
            reason = "missing_or_invalid_eval_loss"
        elif model_type == "unknown":
            reason = "missing_model_type"
        elif optimizer == "unknown":
            reason = "missing_optimizer"
        elif weight_init == "unknown":
            reason = "missing_weight_init"
        elif val_loss is None:
            reason = "missing_val_loss"
        elif model_type in {"mlp", "deep_mlp"} and hidden_activation == "none":
            reason = "missing_hidden_activation"
        elif model_type in {"mlp", "deep_mlp"} and eval_loss == 0.0 and val_loss == 0.0:
            reason = "suspicious_zero_eval_and_val_loss"
        elif is_legacy_or_malformed_name(experiment):
            reason = "legacy_or_malformed_experiment_name"

        if reason is None:
            valid.append(enriched)
        else:
            enriched["_exclude_reason"] = reason
            excluded.append(enriched)

    valid.sort(key=lambda r: r["_eval_loss"])
    excluded.sort(key=lambda r: (r["_exclude_reason"], r.get("experiment", "")))
    return valid, excluded


def config_path_from_experiment_name(experiment_name: str) -> Path:
    return CONFIGS_DIR / f"{experiment_name}.cfg"


def promote_best_config() -> dict:
    rows = read_rows(SUMMARY_CSV)
    valid_rows, excluded_rows = classify_rows(rows)

    if not valid_rows:
        raise RuntimeError("No valid comparable experiment rows were found. Cannot promote a default config.")

    best = valid_rows[0]
    experiment_name = str(best.get("experiment", "")).strip()

    if not experiment_name:
        raise RuntimeError("Best valid row is missing an experiment name.")

    source_config = config_path_from_experiment_name(experiment_name)
    if not source_config.exists():
        raise FileNotFoundError(f"Winning config file does not exist: {source_config}")

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_config, PROMOTED_CONFIG)

    promotion_record = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "promoted_experiment": experiment_name,
        "source_config": str(source_config.relative_to(ROOT)),
        "promoted_config": str(PROMOTED_CONFIG.relative_to(ROOT)),
        "eval_loss": best.get("eval_loss"),
        "train_loss": best.get("train_loss"),
        "val_loss": best.get("val_loss"),
        "model_type": best.get("model_type"),
        "hidden_layers": best.get("hidden_layers"),
        "hidden_activation": best.get("hidden_activation"),
        "optimizer": best.get("optimizer"),
        "weight_init": best.get("weight_init"),
        "valid_run_count": len(valid_rows),
        "excluded_run_count": len(excluded_rows),
    }

    PROMOTION_JSON.write_text(json.dumps(promotion_record, indent=2), encoding="utf-8")
    return promotion_record


def main() -> int:
    try:
        record = promote_best_config()
    except Exception as exc:
        print(f"Promotion failed: {exc}")
        return 1

    print("Promoted best config successfully.")
    print(f"Experiment: {record['promoted_experiment']}")
    print(f"Source: {record['source_config']}")
    print(f"Promoted to: {record['promoted_config']}")
    print(f"Eval loss: {record['eval_loss']}")
    print(f"Optimizer: {record['optimizer']}")
    print(f"Weight init: {record['weight_init']}")
    print(f"Metadata written to: {PROMOTION_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())