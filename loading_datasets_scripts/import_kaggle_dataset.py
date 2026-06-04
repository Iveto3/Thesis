import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_JSON_PATH = Path("input.json")
DATASET_DIR = Path("kaggle_code_optimisation/code-optimization")

UNOPTIMIZED_DIR = DATASET_DIR / "unoptimized"
OPTIMIZED_DIR = DATASET_DIR / "optimized"

OUTPUT_JSON_PATH = Path("input.json")

# True = remove your old code_optimization examples and replace them with Kaggle examples
# False = keep old code_optimization examples and append Kaggle examples
REPLACE_EXISTING_CODE_TASKS = True


def load_existing_tasks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
            print(f"Skipping {unoptimized_file.name}: no matching optimized file")
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
    existing_tasks = load_existing_tasks(INPUT_JSON_PATH)

    if REPLACE_EXISTING_CODE_TASKS:
        existing_tasks = [
            task for task in existing_tasks
            if task.get("task_type") != "code_optimization"
        ]

    code_tasks = build_code_tasks()

    updated_tasks = existing_tasks + code_tasks

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(updated_tasks, file, indent=2, ensure_ascii=False)

    print(f"Imported {len(code_tasks)} Kaggle code optimization examples.")
    print(f"Saved updated tasks to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()