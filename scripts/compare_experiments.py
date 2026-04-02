import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "metrics" / "archive"
RESULTS_DIR = ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_experiments() -> list[dict]:
    experiments = {}

    for path in ARCHIVE_DIR.glob("*_train_metrics.json"):
        name = path.name.replace("_train_metrics.json", "")
        experiments.setdefault(name, {})
        experiments[name]["train"] = load_json(path)

    for path in ARCHIVE_DIR.glob("*_eval_metrics.json"):
        name = path.name.replace("_eval_metrics.json", "")
        experiments.setdefault(name, {})
        experiments[name]["eval"] = load_json(path)

    rows = []
    for name, data in experiments.items():
        train = data.get("train", {})
        evalm = data.get("eval", {})

        rows.append({
            "experiment": name,
            "model_type": train.get("model_type"),
            "hidden_dim": train.get("hidden_dim"),
            "hidden_layers": train.get("hidden_layers"),
            "hidden_activation": train.get("hidden_activation"),
            "num_layers": train.get("num_layers"),
            "epochs": train.get("epochs"),
            "learning_rate": train.get("learning_rate"),
            "lr_schedule": train.get("lr_schedule"),
            "lr_step_size": train.get("lr_step_size"),
            "lr_decay": train.get("lr_decay"),
            "l2_lambda": train.get("l2_lambda"),
            "final_learning_rate": train.get("final_learning_rate"),
            "batch_size": train.get("batch_size"),
            "train_loss": train.get("train_loss"),
            "val_loss": train.get("val_loss"),
            "eval_loss": evalm.get("eval_loss"),
            "prediction_x4": evalm.get("prediction_x4"),
            "parameter_count": evalm.get("parameter_count", train.get("parameter_count")),
            "weight_l2_norm": evalm.get("weight_l2_norm", train.get("weight_l2_norm")),
            "max_abs_weight": evalm.get("max_abs_weight", train.get("max_abs_weight")),
            "bias_l2_norm": evalm.get("bias_l2_norm", train.get("bias_l2_norm")),
        })

    rows.sort(key=lambda r: (r["eval_loss"] is None, r["eval_loss"]))
    return rows


def write_summary(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "model_type",
        "hidden_dim",
        "hidden_layers",
        "hidden_activation",
        "num_layers",
        "epochs",
        "learning_rate",
        "lr_schedule",
        "lr_step_size",
        "lr_decay",
        "l2_lambda",
        "final_learning_rate",
        "batch_size",
        "train_loss",
        "val_loss",
        "eval_loss",
        "prediction_x4",
        "parameter_count",
        "weight_l2_norm",
        "max_abs_weight",
        "bias_l2_norm",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No archived experiment metrics found.")
        return

    print("\n=== Experiment Comparison ===")
    for row in rows:
        print(
            f"{row['experiment']}: "
            f"model={row['model_type']}, "
            f"hidden_dim={row['hidden_dim']}, "
            f"hidden_layers={row['hidden_layers']}, "
            f"hidden_activation={row['hidden_activation']}, "
            f"num_layers={row['num_layers']}, "
            f"epochs={row['epochs']}, "
            f"lr={row['learning_rate']}, "
            f"schedule={row['lr_schedule']}, "
            f"final_lr={row['final_learning_rate']}, "
            f"l2={row['l2_lambda']}, "
            f"batch_size={row['batch_size']}, "
            f"train_loss={row['train_loss']}, "
            f"val_loss={row['val_loss']}, "
            f"eval_loss={row['eval_loss']}, "
            f"params={row['parameter_count']}, "
            f"weight_l2={row['weight_l2_norm']}, "
            f"max_abs_weight={row['max_abs_weight']}, "
            f"pred_x4={row['prediction_x4']}"
        )

    best = rows[0]
    print("\nBest experiment:")
    print(
        f"  {best['experiment']} "
        f"(model={best['model_type']}, "
        f"hidden_dim={best['hidden_dim']}, "
        f"hidden_layers={best['hidden_layers']}, "
        f"num_layers={best['num_layers']}, "
        f"eval_loss={best['eval_loss']}, "
        f"train_loss={best['train_loss']}, "
        f"val_loss={best['val_loss']})"
    )
    print(f"\nSummary written to: {SUMMARY_CSV.relative_to(ROOT)}")


def main() -> int:
    rows = collect_experiments()
    write_summary(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())