import json
import sys
from pathlib import Path


def parse_key_value_file(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def parse_dense_block(data: dict, prefix: str) -> dict:
    input_dim = int(data[f"{prefix}_input_dim"])
    output_dim = int(data[f"{prefix}_output_dim"])

    weights = []
    for i in range(input_dim):
        row = []
        for o in range(output_dim):
            row.append(float(data[f"{prefix}_weight_{i}_{o}"]))
        weights.append(row)

    bias = []
    for o in range(output_dim):
        bias.append(float(data[f"{prefix}_bias_{o}"]))

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "weights": weights,
        "bias": bias,
    }


def parse_checkpoint(path: Path) -> dict:
    data = parse_key_value_file(path)
    model_type = data.get("model_type", "dense")

    if model_type == "dense":
        dense = parse_dense_block(data, "dense")
        return {
            "model_type": "linear",
            "architecture": {
                "input_dim": dense["input_dim"],
                "hidden_dim": 0,
                "output_dim": dense["output_dim"],
            },
            "parameters": {
                "weights": dense["weights"],
                "bias": dense["bias"],
            },
        }

    if model_type == "mlp":
        hidden = parse_dense_block(data, "hidden")
        output = parse_dense_block(data, "output")
        return {
            "model_type": "mlp",
            "architecture": {
                "input_dim": hidden["input_dim"],
                "hidden_dim": hidden["output_dim"],
                "output_dim": output["output_dim"],
            },
            "parameters": {
                "hidden_weights": hidden["weights"],
                "hidden_bias": hidden["bias"],
                "output_weights": output["weights"],
                "output_bias": output["bias"],
            },
        }

    raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/export_model_params.py <checkpoint_path> <output_json_path>")
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