import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import t, wilcoxon


SUMMARY_RESULTS_FILE = Path(
    "results/summarization_results.json"
)

CODE_RESULTS_FILE = Path(
    "results/code_optimization_results.json"
)

SUMMARY_ROUGE_BERT_FILE = Path(
    "computed_stats/summarization_rouge_bert_by_example.json"
)

ALL_STATS_FILE = Path(
    "computed_stats/all_stats.json"
)

CHECKPOINT_ROUNDS = [0, 1, 3, 5, 10, 50]
COMPARISON_ROUNDS = [1, 3, 5, 10, 50]

ALPHA = 0.05


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
    """ Writes to a temporary file first, then replaces the target file. """
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


def collect_similarity_scores(
    results: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """ Collects similarity scores by round. """
    scores_by_round = {
        round_num: {}
        for round_num in CHECKPOINT_ROUNDS
    }

    for result in results:
        task_id = result.get("id")

        if not isinstance(task_id, str):
            continue

        single_shot_similarity = result.get(
            "single_shot_similarity"
        )

        if isinstance(
            single_shot_similarity,
            (int, float),
        ):
            scores_by_round[0][task_id] = float(
                single_shot_similarity
            )

        for checkpoint in result.get(
            "checkpoint_results",
            [],
        ):
            if not isinstance(checkpoint, dict):
                continue

            round_num = checkpoint.get("round")
            similarity = checkpoint.get(
                "similarity"
            )

            if (
                round_num in CHECKPOINT_ROUNDS
                and isinstance(
                    similarity,
                    (int, float),
                )
            ):
                scores_by_round[
                    round_num
                ][task_id] = float(
                    similarity
                )

    return scores_by_round


def collect_summary_metric_scores(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[int, dict[str, float]]:
    """ Collects summary metric scores by round. """
    scores_by_round = {
        round_num: {}
        for round_num in CHECKPOINT_ROUNDS
    }

    for row in rows:
        task_id = row.get("id")
        round_num = row.get("round")
        value = row.get(metric_name)

        if (
            isinstance(task_id, str)
            and round_num in CHECKPOINT_ROUNDS
            and isinstance(value, (int, float))
        ):
            scores_by_round[
                round_num
            ][task_id] = float(value)

    return scores_by_round


def calculate_95_ci(
    values: list[float],
) -> tuple[float, float]:
    """
    Calculate a two-sided 95% confidence interval
    for the mean using the t distribution.
    """
    number_of_values = len(values)
    average = mean(values)

    if number_of_values < 2:
        return average, average

    sample_sd = stdev(values)

    standard_error = (
        sample_sd
        / math.sqrt(number_of_values)
    )

    critical_value = t.ppf(
        1 - ALPHA / 2,
        df=number_of_values - 1,
    )

    margin_of_error = (
        critical_value
        * standard_error
    )

    return (
        average - margin_of_error,
        average + margin_of_error,
    )


def calculate_descriptive_statistics(
    task_name: str,
    metric_name: str,
    round_num: int,
    scores_by_id: dict[str, float],
) -> dict[str, Any]:
    """ Calculate the descriptive statistics. """
    values = list(
        scores_by_id.values()
    )

    if not values:
        raise ValueError(
            f"No scores found for "
            f"{task_name}, {metric_name}, "
            f"round {round_num}"
        )

    ci_lower, ci_upper = calculate_95_ci(
        values
    )

    return {
        "task_name": task_name,
        "metric": metric_name,
        "round": round_num,
        "num_examples": len(values),
        "mean": mean(values),
        "sample_standard_deviation": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "confidence_interval_95_lower": (
            ci_lower
        ),
        "confidence_interval_95_upper": (
            ci_upper
        ),
    }


def calculate_paired_wilcoxon(
    task_name: str,
    metric_name: str,
    baseline_scores: dict[str, float],
    comparison_scores: dict[str, float],
    comparison_round: int,
) -> dict[str, Any]:
    """ Calculate the paired Wilcoxon test. """
    baseline_ids = set(
        baseline_scores
    )

    comparison_ids = set(
        comparison_scores
    )

    if baseline_ids != comparison_ids:
        missing_from_comparison = sorted(
            baseline_ids - comparison_ids
        )

        missing_from_baseline = sorted(
            comparison_ids - baseline_ids
        )

        raise ValueError(
            f"Pairing mismatch for "
            f"{task_name}, {metric_name}, "
            f"round {comparison_round}. "
            f"Missing from comparison: "
            f"{missing_from_comparison[:5]}. "
            f"Missing from baseline: "
            f"{missing_from_baseline[:5]}."
        )

    ordered_ids = sorted(
        baseline_ids
    )

    baseline_values = [
        baseline_scores[task_id]
        for task_id in ordered_ids
    ]

    comparison_values = [
        comparison_scores[task_id]
        for task_id in ordered_ids
    ]

    paired_differences = [
        comparison_score - baseline_score
        for baseline_score, comparison_score
        in zip(
            baseline_values,
            comparison_values,
        )
    ]

    difference_ci_lower, difference_ci_upper = (
        calculate_95_ci(
            paired_differences
        )
    )

    if all(
        abs(difference) < 1e-12
        for difference in paired_differences
    ):
        test_statistic = 0.0
        raw_p_value = 1.0

    else:
        test_result = wilcoxon(
            comparison_values,
            baseline_values,
            alternative="two-sided",
            zero_method="pratt",
            method="auto",
        )

        test_statistic = float(
            test_result.statistic
        )

        raw_p_value = float(
            test_result.pvalue
        )

    return {
        "task_name": task_name,
        "metric": metric_name,
        "baseline_round": 0,
        "comparison_round": comparison_round,
        "num_pairs": len(ordered_ids),
        "baseline_mean": mean(
            baseline_values
        ),
        "comparison_mean": mean(
            comparison_values
        ),
        "mean_paired_difference": mean(
            paired_differences
        ),
        "median_paired_difference": median(
            paired_differences
        ),
        "paired_difference_sample_sd": (
            stdev(paired_differences)
            if len(paired_differences) > 1
            else 0.0
        ),
        "paired_difference_ci95_lower": (
            difference_ci_lower
        ),
        "paired_difference_ci95_upper": (
            difference_ci_upper
        ),
        "wilcoxon_statistic": (
            test_statistic
        ),
        "raw_p_value": raw_p_value,
    }


def apply_holm_correction(
    test_rows: list[dict[str, Any]],
) -> None:
    """
    Applies Holm correction separately
    for each task and metric.
    Each family contains the five tests:
    rounds 0 vs 1, 0 vs 3, 0 vs 5,
    0 vs 10, and 0 vs 50.
    """
    families: dict[
        tuple[str, str],
        list[int],
    ] = {}

    for index, row in enumerate(
        test_rows
    ):
        family = (
            row["task_name"],
            row["metric"],
        )

        families.setdefault(
            family,
            [],
        ).append(index)

    for family_indices in families.values():
        ordered_indices = sorted(
            family_indices,
            key=lambda index: test_rows[
                index
            ]["raw_p_value"],
        )

        number_of_tests = len(
            ordered_indices
        )

        previous_adjusted_value = 0.0

        for rank, row_index in enumerate(
            ordered_indices
        ):
            raw_p_value = test_rows[
                row_index
            ]["raw_p_value"]

            adjusted_p_value = (
                number_of_tests - rank
            ) * raw_p_value

            adjusted_p_value = max(
                adjusted_p_value,
                previous_adjusted_value,
            )

            adjusted_p_value = min(
                adjusted_p_value,
                1.0,
            )

            test_rows[row_index][
                "holm_adjusted_p_value"
            ] = adjusted_p_value

            test_rows[row_index][
                "significant_after_holm"
            ] = (
                adjusted_p_value < ALPHA
            )

            previous_adjusted_value = (
                adjusted_p_value
            )


def add_metric_statistics(
    descriptive_rows: list[dict[str, Any]],
    paired_test_rows: list[dict[str, Any]],
    task_name: str,
    metric_name: str,
    scores_by_round: dict[
        int,
        dict[str, float],
    ],
) -> None:
    """ Adds metric statistics. """

    for round_num in CHECKPOINT_ROUNDS:
        descriptive_rows.append(
            calculate_descriptive_statistics(
                task_name=task_name,
                metric_name=metric_name,
                round_num=round_num,
                scores_by_id=scores_by_round[
                    round_num
                ],
            )
        )

    baseline_scores = scores_by_round[0]

    for comparison_round in COMPARISON_ROUNDS:
        paired_test_rows.append(
            calculate_paired_wilcoxon(
                task_name=task_name,
                metric_name=metric_name,
                baseline_scores=(
                    baseline_scores
                ),
                comparison_scores=(
                    scores_by_round[
                        comparison_round
                    ]
                ),
                comparison_round=(
                    comparison_round
                ),
            )
        )


def print_descriptive_statistics(
    rows: list[dict[str, Any]],
) -> None:
    """ Prints the descriptive statistics. """
    print(
        "\n=== DESCRIPTIVE STATISTICS ==="
    )

    print(
        f"{'Task':<20}"
        f"{'Metric':<22}"
        f"{'Round':<8}"
        f"{'N':<7}"
        f"{'Mean':<11}"
        f"{'SD':<11}"
        f"{'95% CI':<25}"
    )

    for row in rows:
        confidence_interval = (
            f"["
            f"{row['confidence_interval_95_lower']:.4f}, "
            f"{row['confidence_interval_95_upper']:.4f}"
            f"]"
        )

        print(
            f"{row['task_name']:<20}"
            f"{row['metric']:<22}"
            f"{row['round']:<8}"
            f"{row['num_examples']:<7}"
            f"{row['mean']:<11.4f}"
            f"{row['sample_standard_deviation']:<11.4f}"
            f"{confidence_interval:<25}"
        )


def print_paired_tests(
    rows: list[dict[str, Any]],
) -> None:
    """ Prints the paired tests. """
    print(
        "\n=== PAIRED WILCOXON TESTS ==="
    )

    print(
        f"{'Task':<20}"
        f"{'Metric':<22}"
        f"{'Comparison':<12}"
        f"{'N':<7}"
        f"{'Mean diff':<12}"
        f"{'Difference 95% CI':<25}"
        f"{'Raw p':<12}"
        f"{'Holm p':<12}"
        f"{'Sig.':<8}"
    )

    for row in rows:
        comparison = (
            f"0 vs "
            f"{row['comparison_round']}"
        )

        confidence_interval = (
            f"["
            f"{row['paired_difference_ci95_lower']:.4f}, "
            f"{row['paired_difference_ci95_upper']:.4f}"
            f"]"
        )

        print(
            f"{row['task_name']:<20}"
            f"{row['metric']:<22}"
            f"{comparison:<12}"
            f"{row['num_pairs']:<7}"
            f"{row['mean_paired_difference']:<12.4f}"
            f"{confidence_interval:<25}"
            f"{row['raw_p_value']:<12.6g}"
            f"{row['holm_adjusted_p_value']:<12.6g}"
            f"{str(row['significant_after_holm']):<8}"
        )


def main() -> None:
    """ Main function. """
    summarization_results = load_json(
        SUMMARY_RESULTS_FILE
    )

    code_results = load_json(
        CODE_RESULTS_FILE
    )

    summary_rouge_bert_rows = load_json(
        SUMMARY_ROUGE_BERT_FILE
    )

    all_stats = load_json(
        ALL_STATS_FILE
    )

    if not isinstance(
        summarization_results,
        list,
    ):
        raise ValueError(
            "Summarization results must "
            "contain a JSON list."
        )

    if not isinstance(
        code_results,
        list,
    ):
        raise ValueError(
            "Code optimization results must "
            "contain a JSON list."
        )

    if not isinstance(
        summary_rouge_bert_rows,
        list,
    ):
        raise ValueError(
            "ROUGE/BERTScore results must "
            "contain a JSON list."
        )

    if not isinstance(
        all_stats,
        dict,
    ):
        raise ValueError(
            "all_stats.json must contain "
            "a JSON object."
        )

    summarization_cosine_scores = (
        collect_similarity_scores(
            summarization_results
        )
    )

    code_cosine_scores = (
        collect_similarity_scores(
            code_results
        )
    )

    summarization_rouge_scores = (
        collect_summary_metric_scores(
            summary_rouge_bert_rows,
            metric_name="rouge_lsum_f1",
        )
    )

    summarization_bertscore_scores = (
        collect_summary_metric_scores(
            summary_rouge_bert_rows,
            metric_name="bertscore_f1",
        )
    )

    descriptive_rows = []
    paired_test_rows = []

    add_metric_statistics(
        descriptive_rows=descriptive_rows,
        paired_test_rows=paired_test_rows,
        task_name="summarization",
        metric_name="cosine_similarity",
        scores_by_round=(
            summarization_cosine_scores
        ),
    )

    add_metric_statistics(
        descriptive_rows=descriptive_rows,
        paired_test_rows=paired_test_rows,
        task_name="summarization",
        metric_name="rouge_lsum_f1",
        scores_by_round=(
            summarization_rouge_scores
        ),
    )

    add_metric_statistics(
        descriptive_rows=descriptive_rows,
        paired_test_rows=paired_test_rows,
        task_name="summarization",
        metric_name="bertscore_f1",
        scores_by_round=(
            summarization_bertscore_scores
        ),
    )

    add_metric_statistics(
        descriptive_rows=descriptive_rows,
        paired_test_rows=paired_test_rows,
        task_name="code_optimization",
        metric_name="cosine_similarity",
        scores_by_round=(
            code_cosine_scores
        ),
    )

    apply_holm_correction(
        paired_test_rows
    )

    all_stats[
        "continuous_metric_descriptive_statistics"
    ] = descriptive_rows

    all_stats[
        "paired_wilcoxon_tests"
    ] = paired_test_rows

    write_json_atomic(
        ALL_STATS_FILE,
        all_stats,
    )

    print_descriptive_statistics(
        descriptive_rows
    )

    print_paired_tests(
        paired_test_rows
    )

    print(
        "\nStatistical results added to:"
    )

    print(ALL_STATS_FILE)


if __name__ == "__main__":
    main()
