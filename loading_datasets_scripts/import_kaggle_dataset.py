import json
from pathlib import Path
from typing import Any, Dict, List


DATASET_DIR = Path("kaggle_code_optimisation/code-optimization")

UNOPTIMIZED_DIR = DATASET_DIR / "unoptimized"
OPTIMIZED_DIR = DATASET_DIR / "optimized"

OUTPUT_JSON_PATH = Path("input/code_optimization_input.json")


def read_code_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_code_tasks() -> List[Dict[str, Any]]:
    if not UNOPTIMIZED_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {UNOPTIMIZED_DIR}")

    if not OPTIMIZED_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {OPTIMIZED_DIR}")

    unoptimized_files = sorted(
        file for file in UNOPTIMIZED_DIR.iterdir()
        if file.is_file()
    )

    code_tasks = []

    for index, unoptimized_file in enumerate(unoptimized_files, start=1):
        optimized_file = OPTIMIZED_DIR / unoptimized_file.name

        if not optimized_file.exists():
            print(f"Skipping {unoptimized_file.name}:")
            print("no matching optimized file")
            continue

        original_code = read_code_file(unoptimized_file)
        optimized_code = read_code_file(optimized_file)

        task = {
            "id": f"code_{index:03d}",
            "task_type": "code_optimization",
            "input": {
                "code": original_code
            },
            "constraints": {
                "goal": "optimize, improve readability and performance"
            },
            "reference_output": {
                "optimized_code": optimized_code
            }
        }
        code_tasks.append(task)
    return code_tasks


def main() -> None:
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    code_tasks = build_code_tasks()

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(code_tasks, file, indent=2, ensure_ascii=False)

    print(f"Imported {len(code_tasks)} Kaggle code optimization examples.")
    print(f"Saved code optimization tasks to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
