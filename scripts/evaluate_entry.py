import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def resolve_evaluate_executable() -> Path:
    if sys.platform.startswith("win"):
        raise RuntimeError(
            "Native Windows build is disabled for now. "
            "Run evaluation through Docker/Linux using build-docker/apps/evaluate_app."
        )
    return ROOT / "build-docker" / "apps" / "evaluate_app"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_entry.py <config_path>")
        return 1

    config_path = sys.argv[1]

    try:
        exe_path = resolve_evaluate_executable()
    except RuntimeError as exc:
        print(exc)
        return 1

    if not exe_path.exists():
        print(f"Evaluate executable not found: {exe_path}")
        return 1

    result = subprocess.run([str(exe_path), config_path], cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())