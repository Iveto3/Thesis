import json
from pathlib import Path
from statistics import mean
from typing import Any

from metrics import compute_bertscore_batch, compute_rouge_lsum


SUMMARY_INPUT_FILE = Path(
    "input/summarization_dataset_input.json"
)

SUMMARY_RESULTS_FILE = Path(
    "results/summarization_results.json"
)

ALL_STATS_FILE = Path(
    "computed_stats/all_stats.json"
)

PER_EXAMPLE_FILE = Path(
    "computed_stats/summarization_rouge_bert_by_example.json"
)

CHECKPOINT_ROUNDS = [0, 1, 3, 5, 10, 50]

BERT_BATCH_SIZE = 8


def load_json(file_path: Path) -> Any:
    """ Loads a JSON file. """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json_atomic(
    file_path: Path,
    data: Any,
) -> None:
    """
    Write to a temporary file first, then replace the target file.
    This reduces the risk of corrupting the JSON file if the process
    stops while writing.
    """
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = file_path.with_suffix(
        file_path.suffix + ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(file_path)


def build_reference_map(
    input_tasks: list[dict[str, Any]],
) -> dict[str, str]:
    """ Builds a map from task ID to reference summary. """

    references = {}

    for task in input_tasks:
        task_id = task.get("id")
        reference_output = task.get(
            "reference_output"
        )

        if not isinstance(task_id, str):
            continue

        if not isinstance(
            reference_output,
            dict,
        ):
            continue

        reference = reference_output.get(
            "summary"
        )

        if (
            isinstance(reference, str)
            and reference.strip()
        ):
            references[task_id] = reference

    return references


def collect_examples_by_round(
    results: list[dict[str, Any]],
    references: dict[str, str],
) -> dict[int, list[dict[str, str]]]:
    """ Collects examples by round. """
    examples_by_round = {
        round_num: []
        for round_num in CHECKPOINT_ROUNDS
    }

    for result in results:
        task_id = result.get("id")

        if not isinstance(task_id, str):
            continue

        reference = references.get(task_id)

        if reference is None:
            raise ValueError(
                f"No reference summary found for {task_id}"
            )

        single_shot_answer = result.get(
            "single_shot_answer"
        )

        if (
            not isinstance(
                single_shot_answer,
                str,
            )
            or not single_shot_answer.strip()
        ):
            raise ValueError(
                f"No single-shot answer found for {task_id}"
            )

        examples_by_round[0].append(
            {
                "id": task_id,
                "answer": single_shot_answer,
                "reference": reference,
            }
        )

        checkpoint_answers = {}

        for checkpoint in result.get(
            "checkpoint_results",
            [],
        ):
            if not isinstance(checkpoint, dict):
                continue

            round_num = checkpoint.get("round")
            answer = checkpoint.get("answer")

            if (
                isinstance(round_num, int)
                and isinstance(answer, str)
                and answer.strip()
            ):
                checkpoint_answers[
                    round_num
                ] = answer

        for round_num in CHECKPOINT_ROUNDS[1:]:
            answer = checkpoint_answers.get(
                round_num
            )

            if answer is None:
                raise ValueError(
                    f"No answer found for {task_id} "
                    f"at round {round_num}"
                )

            examples_by_round[
                round_num
            ].append(
                {
                    "id": task_id,
                    "answer": answer,
                    "reference": reference,
                }
            )

    return examples_by_round


def main() -> None:
    """ Main function. """
    input_tasks = load_json(
        SUMMARY_INPUT_FILE
    )

    summarization_results = load_json(
        SUMMARY_RESULTS_FILE
    )

    all_stats = load_json(
        ALL_STATS_FILE
    )

    if not isinstance(input_tasks, list):
        raise ValueError(
            "The summarization input file must contain a JSON list."
        )

    if not isinstance(
        summarization_results,
        list,
    ):
        raise ValueError(
            "The summarization results file must contain a JSON list."
        )

    if not isinstance(all_stats, dict):
        raise ValueError(
            "all_stats.json must contain a JSON object."
        )

    references = build_reference_map(
        input_tasks
    )

    examples_by_round = (
        collect_examples_by_round(
            summarization_results,
            references,
        )
    )

    summary_rows = []
    per_example_rows = []

    for round_num in CHECKPOINT_ROUNDS:
        examples = examples_by_round[
            round_num
        ]

        ids = [
            example["id"]
            for example in examples
        ]

        responses = [
            example["answer"]
            for example in examples
        ]

        round_references = [
            example["reference"]
            for example in examples
        ]

        print(
            f"\nRound {round_num}: "
            f"{len(examples)} examples",
            flush=True,
        )

        print(
            "Computing ROUGE-Lsum...",
            flush=True,
        )

        rouge_scores = [
            compute_rouge_lsum(
                response,
                reference,
            )
            for response, reference in zip(
                responses,
                round_references,
            )
        ]

        print(
            "Computing BERTScore...",
            flush=True,
        )

        bert_scores = compute_bertscore_batch(
            responses=responses,
            references=round_references,
            batch_size=BERT_BATCH_SIZE,
        )

        for (
            task_id,
            rouge_score,
            bert_score,
        ) in zip(
            ids,
            rouge_scores,
            bert_scores,
        ):
            per_example_rows.append(
                {
                    "id": task_id,
                    "round": round_num,
                    "rouge_lsum_f1": (
                        rouge_score
                    ),
                    "bertscore_precision": (
                        bert_score[
                            "bertscore_precision"
                        ]
                    ),
                    "bertscore_recall": (
                        bert_score[
                            "bertscore_recall"
                        ]
                    ),
                    "bertscore_f1": (
                        bert_score[
                            "bertscore_f1"
                        ]
                    ),
                }
            )

        round_stats = {
            "round": round_num,
            "num_examples": len(examples),
            "average_rouge_lsum_f1": mean(
                rouge_scores
            ),
            "average_bertscore_precision": mean(
                score[
                    "bertscore_precision"
                ]
                for score in bert_scores
            ),
            "average_bertscore_recall": mean(
                score[
                    "bertscore_recall"
                ]
                for score in bert_scores
            ),
            "average_bertscore_f1": mean(
                score[
                    "bertscore_f1"
                ]
                for score in bert_scores
            ),
        }

        summary_rows.append(round_stats)

        all_stats[
            "summarization_rouge_bert_stats"
        ] = summary_rows

        write_json_atomic(
            ALL_STATS_FILE,
            all_stats,
        )

        write_json_atomic(
            PER_EXAMPLE_FILE,
            per_example_rows,
        )

        print(
            f"Finished round {round_num}: "
            f"ROUGE-Lsum="
            f"{round_stats['average_rouge_lsum_f1']:.4f}, "
            f"BERTScore F1="
            f"{round_stats['average_bertscore_f1']:.4f}",
            flush=True,
        )

    print(
        "\n=== ROUGE-LSUM AND BERTSCORE ==="
    )

    print(
        f"{'Round':<8}"
        f"{'N':<8}"
        f"{'ROUGE-Lsum':<16}"
        f"{'BERT-P':<12}"
        f"{'BERT-R':<12}"
        f"{'BERT-F1':<12}"
    )

    for row in summary_rows:
        print(
            f"{row['round']:<8}"
            f"{row['num_examples']:<8}"
            f"{row['average_rouge_lsum_f1']:<16.4f}"
            f"{row['average_bertscore_precision']:<12.4f}"
            f"{row['average_bertscore_recall']:<12.4f}"
            f"{row['average_bertscore_f1']:<12.4f}"
        )

    print(
        f"\nROUGE and BERT averages added to: "
        f"{ALL_STATS_FILE}"
    )

    print(
        f"Per-example scores saved to: "
        f"{PER_EXAMPLE_FILE}"
    )


if __name__ == "__main__":
    main()
