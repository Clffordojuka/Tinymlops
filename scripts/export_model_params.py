import json
import sys
from pathlib import Path


def parse_checkpoint(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value

    input_dim = int(data["input_dim"])
    output_dim = int(data["output_dim"])

    weights = []
    for i in range(input_dim):
        row = []
        for o in range(output_dim):
            row.append(float(data[f"weight_{i}_{o}"]))
        weights.append(row)

    bias = []
    for o in range(output_dim):
        bias.append(float(data[f"bias_{o}"]))

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "weights": weights,
        "bias": bias,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/export_model_params.py <checkpoint_path> <output_json>")
        return 1

    checkpoint_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    model_params = parse_checkpoint(checkpoint_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model_params, indent=2), encoding="utf-8")

    print(f"Exported model parameters to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())