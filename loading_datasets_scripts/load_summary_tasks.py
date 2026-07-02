import json
from itertools import islice
from pathlib import Path
from datasets import load_dataset


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def main() -> None:
    num_examples = 1000
    output_file = Path("input/summarization_dataset_input.json")
    output_file.parent.mkdir(exist_ok=True)

    dataset = load_dataset(
        "abisee/cnn_dailymail",
        "1.0.0",
        split="train",
        streaming=True,
    )

    summary_tasks = []

    for i, example in enumerate(islice(dataset, num_examples)):
        task = {
            "id": f"summary_{i:05d}",
            "task_type": "summarization",
            "input": {
                "text": clean_text(example["article"])
            },
            "constraints": {
                "goal": "summarize by keeping the main points and no extra details",
                "max_words": 150,
            },
            "reference_output": {
                "summary": clean_text(example["highlights"])
            },
        }

        summary_tasks.append(task)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary_tasks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(summary_tasks)} summarization tasks to {output_file}.")


if __name__ == "__main__":
    main()
