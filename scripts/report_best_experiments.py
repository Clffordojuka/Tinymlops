import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
REPORT_MD = RESULTS_DIR / "benchmark_report.md"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    return rows


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


def valid_rows(rows: list[dict]) -> list[dict]:
    cleaned = []

    for row in rows:
        eval_loss = parse_float(row.get("eval_loss"))
        train_loss = parse_float(row.get("train_loss"))
        val_loss = parse_float(row.get("val_loss"))

        cleaned.append({
            **row,
            "_eval_loss": eval_loss,
            "_train_loss": train_loss,
            "_val_loss": val_loss,
            "_model_type": normalize_text(row.get("model_type")),
            "_hidden_activation": normalize_text(row.get("hidden_activation"), "none"),
            "_optimizer": normalize_text(row.get("optimizer")),
            "_weight_init": normalize_text(row.get("weight_init")),
        })

    cleaned = [r for r in cleaned if r["_eval_loss"] is not None and r["_eval_loss"] >= 0.0]
    cleaned.sort(key=lambda r: r["_eval_loss"])
    return cleaned


def best_by_key(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    winners = {}
    for group_name, group_rows in groups.items():
        group_rows.sort(key=lambda r: r["_eval_loss"])
        winners[group_name] = group_rows[0]

    return winners


def fmt_row(row: dict) -> str:
    return (
        f"{row.get('experiment')} | "
        f"model={row.get('model_type')} | "
        f"hidden_layers={row.get('hidden_layers')} | "
        f"activation={row.get('hidden_activation')} | "
        f"optimizer={row.get('optimizer')} | "
        f"weight_init={row.get('weight_init')} | "
        f"eval_loss={row.get('eval_loss')} | "
        f"train_loss={row.get('train_loss')} | "
        f"val_loss={row.get('val_loss')}"
    )


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Rank | Experiment | Model | Hidden Layers | Activation | Optimizer | Weight Init | Eval Loss | Train Loss | Val Loss |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|",
    ]

    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row.get('experiment')} | {row.get('model_type')} | "
            f"{row.get('hidden_layers')} | {row.get('hidden_activation')} | "
            f"{row.get('optimizer')} | {row.get('weight_init')} | "
            f"{row.get('eval_loss')} | {row.get('train_loss')} | {row.get('val_loss')} |"
        )

    return "\n".join(lines)


def write_report(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        REPORT_MD.write_text(
            "# Benchmark Report\n\nNo valid experiment rows found in experiment_summary.csv.\n",
            encoding="utf-8",
        )
        return REPORT_MD

    best_overall = rows[0]
    top5 = rows[:5]

    best_models = best_by_key(rows, "_model_type")
    best_activations = best_by_key(rows, "_hidden_activation")
    best_optimizers = best_by_key(rows, "_optimizer")
    best_inits = best_by_key(rows, "_weight_init")

    report_lines = [
        "# Benchmark Report",
        "",
        "## Best Overall Experiment",
        "",
        f"- **Experiment:** {best_overall.get('experiment')}",
        f"- **Model Type:** {best_overall.get('model_type')}",
        f"- **Hidden Layers:** {best_overall.get('hidden_layers')}",
        f"- **Hidden Activation:** {best_overall.get('hidden_activation')}",
        f"- **Optimizer:** {best_overall.get('optimizer')}",
        f"- **Weight Init:** {best_overall.get('weight_init')}",
        f"- **Eval Loss:** {best_overall.get('eval_loss')}",
        f"- **Train Loss:** {best_overall.get('train_loss')}",
        f"- **Val Loss:** {best_overall.get('val_loss')}",
        "",
        "## Top 5 Experiments",
        "",
        markdown_table(top5),
        "",
        "## Best by Model Type",
        "",
    ]

    for key in sorted(best_models):
        report_lines.append(f"- **{key}:** {fmt_row(best_models[key])}")

    report_lines.extend([
        "",
        "## Best by Hidden Activation",
        "",
    ])

    for key in sorted(best_activations):
        report_lines.append(f"- **{key}:** {fmt_row(best_activations[key])}")

    report_lines.extend([
        "",
        "## Best by Optimizer",
        "",
    ])

    for key in sorted(best_optimizers):
        report_lines.append(f"- **{key}:** {fmt_row(best_optimizers[key])}")

    report_lines.extend([
        "",
        "## Best by Weight Initialization",
        "",
    ])

    for key in sorted(best_inits):
        report_lines.append(f"- **{key}:** {fmt_row(best_inits[key])}")

    report_lines.extend([
        "",
        "## Recommended Default",
        "",
        f"The current recommended default configuration is **{best_overall.get('experiment')}**, "
        f"because it achieved the lowest recorded eval loss of **{best_overall.get('eval_loss')}** "
        f"among the benchmarked runs.",
        "",
    ])

    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return REPORT_MD


def main() -> int:
    rows = read_rows(SUMMARY_CSV)
    ranked_rows = valid_rows(rows)
    report_path = write_report(ranked_rows)
    print(f"Benchmark report written to: {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())