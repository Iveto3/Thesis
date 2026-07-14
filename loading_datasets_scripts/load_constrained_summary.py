import json
from itertools import islice
from pathlib import Path
from datasets import load_dataset


def clean_text(text: str) -> str:
    """ Removes newlines from text. """
    return " ".join(text.replace("\n", " ").split())


def main() -> None:
    """ Main function. """
    num_examples = 100
    output_file = Path("input/constrained_summary_input.json")
    output_file.parent.mkdir(exist_ok=True)

    dataset = load_dataset(
        "abisee/cnn_dailymail",
        "1.0.0",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    constrained_tasks = []

    for i, example in enumerate(islice(dataset, num_examples)):
        task = {
            "id": f"constrained_summary_{i:05d}",
            "task_type": "constrained_summarization",
            "input": {
                "text": clean_text(example["article"])
            },
            "constraints": {
                "goal": "summarize the article while keeping the main news points and no extra details",
                "max_words": 150,
                "tone": "news reporter",
                "style": "rhyming but factual",
                "must_include": [
                    "main event",
                    "key people or organizations",
                    "main outcome"
                ],
                "must_avoid": [
                    "unsupported claims",
                    "extra details",
                    "jokes"
                ]
            },
            "reference_output": None
        }

        constrained_tasks.append(task)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(constrained_tasks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(constrained_tasks)} constrained summarization tasks to {output_file}.")


if __name__ == "__main__":
    main()
