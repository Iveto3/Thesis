import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


SUMMARY_RESULTS_FILE = "results/summarization_results.json"
CODE_RESULTS_FILE = "results/code_optimization_results.json"
CONSTRAINED_RESULTS_FILE = "results/constrained_summary_results.json"

OUTPUT_DIR = Path("computed_stats")

GENERAL_EVAL_CATEGORIES = [
    "goal",
    "faithfulness",
    "coverage",
    "tone",
    "style",
    "length",
    "entities",
]

PARTIAL_LENGTH_LIMIT = 200


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON file."""

    path = Path(file_path)

    if not path.exists():
        print(f"Missing file: {file_path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Writes a JSON file."""

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_mean(values: List[float]) -> Optional[float]:
    """Returns the mean of a list of values, or None if the list is empty."""

    if not values:
        return None
    return mean(values)


def fmt(value: Optional[float], decimals: int = 4) -> str:
    """Returns a string representation of a float,
    or "N/A" if the value is None."""

    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def count_words(text: str) -> int:
    """Returns the number of words in a string."""
    return len(text.split())


def compute_manual_length_evaluation(
    answer: str,
    max_words: int = 150,
) -> Dict[str, Any]:
    """
    Manual length scoring:
    score 2 = answer is at or below max_words
    score 1 = answer is above max_words but at or below PARTIAL_LENGTH_LIMIT
    score 0 = answer is above PARTIAL_LENGTH_LIMIT
    """
    word_count = count_words(answer)

    if word_count <= max_words:
        score = 2
    elif word_count <= PARTIAL_LENGTH_LIMIT:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "satisfied": score == 2,
        "word_count": word_count,
    }


def normalize_evaluation_with_manual_length(
    evaluation: Dict[str, Any],
    answer: str,
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Copies the LLM-judge evaluation but replaces the length category
    with a manual length score based on the stored answer.
    """
    normalized = dict(evaluation)

    max_words = constraints.get("max_words", 150)

    if not isinstance(max_words, int):
        max_words = 150

    normalized["length"] = compute_manual_length_evaluation(
        answer=answer,
        max_words=max_words,
    )

    return normalized


def collect_similarity_by_round(
    results: List[Dict[str, Any]]
) -> Dict[int, List[float]]:
    values_by_round: Dict[int, List[float]] = {}
    """Collects similarity values by round."""

    for result in results:
        single_shot_similarity = result.get("single_shot_similarity")

        if isinstance(single_shot_similarity, (int, float)):
            values_by_round.setdefault(0, []).append(
                float(single_shot_similarity))

        for checkpoint in result.get("checkpoint_results", []):
            round_num = checkpoint.get("round")
            similarity = checkpoint.get("similarity")

            if isinstance(round_num, int) and isinstance(similarity, (
                                                            int, float)):
                values_by_round.setdefault(
                    round_num, []).append(float(similarity))

    return values_by_round


def compute_similarity_stats(
    results: List[Dict[str, Any]],
    task_name: str,
) -> List[Dict[str, Any]]:
    """
    Computes only average similarity per checkpoint.
    """
    values_by_round = collect_similarity_by_round(results)
    rows = []

    for round_num in sorted(values_by_round):
        values = values_by_round[round_num]

        rows.append(
            {
                "task_name": task_name,
                "round": round_num,
                "num_examples": len(values),
                "average_similarity": safe_mean(values),
            }
        )

    return rows


def print_similarity_stats(rows: List[Dict[str, Any]], title: str) -> None:
    """Prints similarity stats."""
    print(f"\n=== {title} ===")

    if not rows:
        print("No similarity stats found.")
        return

    print(f"{'Round':<8} {'N':<6} {'Avg Sim':<10}")

    for row in rows:
        print(
            f"{row['round']:<8} "
            f"{row['num_examples']:<6} "
            f"{fmt(row.get('average_similarity')):<10}"
        )


def collect_constrained_evaluations(
    results: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Collects constrained-summary evaluations by round.
    The stored LLM-judge evaluation is used for all categories except length.
    """

    evaluations_by_round: Dict[int, List[Dict[str, Any]]] = {}

    for result in results:
        constraints = result.get("constraints", {})

        if not isinstance(constraints, dict):
            constraints = {}

        single_eval = result.get("single_shot_constraint_evaluation")
        single_answer = result.get("single_shot_answer", "")

        if isinstance(single_eval, dict) and not single_eval.get(
             "parse_error"):
            normalized_eval = normalize_evaluation_with_manual_length(
                evaluation=single_eval,
                answer=single_answer,
                constraints=constraints,
            )
            evaluations_by_round.setdefault(0, []).append(normalized_eval)

        for checkpoint in result.get("checkpoint_results", []):
            round_num = checkpoint.get("round")
            evaluation = checkpoint.get("constraint_evaluation")
            answer = checkpoint.get("answer", "")

            if (
                isinstance(round_num, int)
                and isinstance(evaluation, dict)
                and not evaluation.get("parse_error")
            ):
                normalized_eval = normalize_evaluation_with_manual_length(
                    evaluation=evaluation,
                    answer=answer,
                    constraints=constraints,
                )
                evaluations_by_round.setdefault(
                    round_num, []).append(normalized_eval)

    return evaluations_by_round


def get_category_score(
    evaluation: Dict[str, Any],
    category: str,
) -> Optional[float]:
    """
    Returns only valid scores: 0, 1, or 2.
    Any invalid LLM-judge score is ignored.
    """
    value = evaluation.get(category)

    if not isinstance(value, dict):
        return None

    score = value.get("score")

    if isinstance(score, bool):
        return None

    if isinstance(score, (int, float)) and score in [0, 1, 2]:
        return float(score)

    return None


def get_category_satisfied(
    evaluation: Dict[str, Any],
    category: str,
) -> Optional[float]:
    """
    Manual success/fail rule:
    score 2 = pass/success
    score 0 or 1 = fail/not fully satisfied
    """
    score = get_category_score(evaluation, category)

    if score is None:
        return None

    return 1.0 if score == 2 else 0.0


def get_items(
    evaluation: Dict[str, Any],
    field_name: str,
) -> List[Dict[str, Any]]:
    items = evaluation.get(field_name, [])
    """
    Returns only valid items.
    Any invalid item is ignored.
    """

    if not isinstance(items, list):
        return []

    return [item for item in items if isinstance(item, dict)]


def get_item_scores(
    evaluation: Dict[str, Any],
    field_name: str,
) -> List[float]:
    """
    Returns only valid scores: 0, 1, or 2.
    Any invalid LLM-judge score is ignored.
    """

    scores = []

    for item in get_items(evaluation, field_name):
        score = item.get("score")

        if isinstance(score, bool):
            continue

        if isinstance(score, (int, float)) and score in [0, 1, 2]:
            scores.append(float(score))

    return scores


def get_expected_total(
    evaluation: Dict[str, Any],
    total_field_name: str,
    fallback: int,
) -> int:
    """
    Returns total if valid, otherwise returns fallback.
    """

    total = evaluation.get(total_field_name)

    if isinstance(total, (int, float)) and total >= 0:
        return int(total)

    return fallback


def count_successes_from_scores(scores: List[float]) -> int:
    """Counts only successful LLM-judge scores of 2."""
    return sum(1 for score in scores if score == 2)


def compute_manual_completion_ratio(
     evaluation: Dict[str, Any]) -> Optional[float]:
    """
    Computes completion from numeric scores only.
    Missing category scores count as 0.
    Missing must_include/must_avoid items count as 0 by using expected totals
    in the denominator.
    """
    category_scores = []

    for category in GENERAL_EVAL_CATEGORIES:
        score = get_category_score(evaluation, category)
        category_scores.append(score if score is not None else 0.0)

    must_include_scores = get_item_scores(evaluation, "must_include_items")
    must_avoid_scores = get_item_scores(evaluation, "must_avoid_items")

    must_include_total = get_expected_total(
        evaluation,
        "must_include_total",
        fallback=len(must_include_scores),
    )
    must_avoid_total = get_expected_total(
        evaluation,
        "must_avoid_total",
        fallback=len(must_avoid_scores),
    )

    must_include_total = max(must_include_total, len(must_include_scores))
    must_avoid_total = max(must_avoid_total, len(must_avoid_scores))

    total_score = (
        sum(category_scores)
        + sum(must_include_scores)
        + sum(must_avoid_scores)
    )

    max_possible_score = 2 * (
        len(GENERAL_EVAL_CATEGORIES)
        + must_include_total
        + must_avoid_total
    )

    if max_possible_score == 0:
        return None

    return total_score / max_possible_score


def compute_manual_overall_pass(evaluation: Dict[str, Any]) -> bool:
    """
    Computes overall pass rate.
    """
    for category in GENERAL_EVAL_CATEGORIES:
        score = get_category_score(evaluation, category)

        if score != 2:
            return False

    must_include_scores = get_item_scores(evaluation, "must_include_items")
    must_avoid_scores = get_item_scores(evaluation, "must_avoid_items")

    must_include_total = get_expected_total(
        evaluation,
        "must_include_total",
        fallback=len(must_include_scores),
    )
    must_avoid_total = get_expected_total(
        evaluation,
        "must_avoid_total",
        fallback=len(must_avoid_scores),
    )

    if len(must_include_scores) != must_include_total:
        return False

    if len(must_avoid_scores) != must_avoid_total:
        return False

    if any(score != 2 for score in must_include_scores):
        return False

    if any(score != 2 for score in must_avoid_scores):
        return False

    return True


def compute_constrained_overall_stats(
    evaluations_by_round: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Computes constrained overall stats.
    """
    rows = []

    for round_num in sorted(evaluations_by_round):
        evaluations = evaluations_by_round[round_num]

        completion_values = []
        overall_pass_values = []
        must_include_ratios = []
        must_avoid_ratios = []

        for evaluation in evaluations:
            manual_completion = compute_manual_completion_ratio(evaluation)

            if manual_completion is not None:
                completion_values.append(manual_completion)

            manual_overall_pass = compute_manual_overall_pass(evaluation)
            overall_pass_values.append(1.0 if manual_overall_pass else 0.0)

            must_include_scores = get_item_scores(
                evaluation, "must_include_items")
            must_avoid_scores = get_item_scores(evaluation, "must_avoid_items")

            must_include_total = get_expected_total(
                evaluation,
                "must_include_total",
                fallback=len(must_include_scores),
            )
            must_avoid_total = get_expected_total(
                evaluation,
                "must_avoid_total",
                fallback=len(must_avoid_scores),
            )

            if must_include_total > 0:
                mi_success_count = count_successes_from_scores(
                    must_include_scores)
                must_include_ratios.append(
                    mi_success_count / must_include_total)

            if must_avoid_total > 0:
                ma_success_count = count_successes_from_scores(
                    must_avoid_scores)
                must_avoid_ratios.append(ma_success_count / must_avoid_total)

        rows.append(
            {
                "round": round_num,
                "num_examples": len(evaluations),
                "average_constraint_completion_ratio": safe_mean(
                    completion_values),
                "overall_pass_rate": safe_mean(overall_pass_values),
                "average_must_include_ratio": safe_mean(must_include_ratios),
                "average_must_avoid_ratio": safe_mean(must_avoid_ratios),
            }
        )

    return rows


def compute_constrained_category_stats(
    evaluations_by_round: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Computes constrained category stats.
    """
    rows = []

    for round_num in sorted(evaluations_by_round):
        evaluations = evaluations_by_round[round_num]

        row = {
            "round": round_num,
            "num_examples": len(evaluations),
        }

        for category in GENERAL_EVAL_CATEGORIES:
            pass_values = []

            for evaluation in evaluations:
                passed = get_category_satisfied(evaluation, category)

                if passed is not None:
                    pass_values.append(passed)

            row[f"{category}_pass_rate"] = safe_mean(pass_values)

        rows.append(row)

    return rows


def print_constrained_overall_stats(rows: List[Dict[str, Any]]) -> None:
    """ Prints constrained overall stats. """
    print("\n=== CONSTRAINED SUMMARY OVERALL STATS ===")

    if not rows:
        print("No constrained summary overall stats found.")
        return

    print(
        f"{'Round':<8} "
        f"{'N':<6} "
        f"{'Completion':<12} "
        f"{'Pass Rate':<12} "
        f"{'MustInc Ratio':<14} "
        f"{'MustAvoid Ratio':<15}"
    )

    for row in rows:
        print(
            f"{row['round']:<8} "
            f"{row['num_examples']:<6} "
            f"{fmt(row.get('average_constraint_completion_ratio')):<12} "
            f"{fmt(row.get('overall_pass_rate')):<12} "
            f"{fmt(row.get('average_must_include_ratio')):<14} "
            f"{fmt(row.get('average_must_avoid_ratio')):<15}"
        )


def print_constrained_category_stats(rows: List[Dict[str, Any]]) -> None:
    """ Prints constrained category stats. """
    print("\n=== CONSTRAINED SUMMARY CATEGORY PASS RATES ===")

    if not rows:
        print("No constrained summary category stats found.")
        return

    for row in rows:
        print(f"\nRound {row['round']} | N={row['num_examples']}")

        for category in GENERAL_EVAL_CATEGORIES:
            pass_rate = row.get(f"{category}_pass_rate")

            print(
                f"  {category:<14} "
                f"pass_rate={fmt(pass_rate)}"
            )


def main() -> None:
    """ Main function. """
    OUTPUT_DIR.mkdir(exist_ok=True)

    summarization_results = load_json(SUMMARY_RESULTS_FILE)
    code_results = load_json(CODE_RESULTS_FILE)
    constrained_results = load_json(CONSTRAINED_RESULTS_FILE)

    summarization_stats = compute_similarity_stats(
        summarization_results,
        task_name="summarization",
    )

    code_stats = compute_similarity_stats(
        code_results,
        task_name="code_optimization",
    )

    constrained_evaluations = collect_constrained_evaluations(
        constrained_results)

    constrained_overall_stats = compute_constrained_overall_stats(
        constrained_evaluations
    )

    constrained_category_stats = compute_constrained_category_stats(
        constrained_evaluations
    )

    all_stats = {
        "summarization_similarity_stats": summarization_stats,
        "code_optimization_similarity_stats": code_stats,
        "constrained_summary_overall_stats": constrained_overall_stats,
        "constrained_summary_category_stats": constrained_category_stats,
    }

    write_json(OUTPUT_DIR / "all_stats.json", all_stats)

    print_similarity_stats(
        summarization_stats,
        title="SUMMARIZATION SIMILARITY STATS",
    )

    print_similarity_stats(
        code_stats,
        title="CODE OPTIMIZATION SIMILARITY STATS",
    )

    print_constrained_overall_stats(constrained_overall_stats)
    print_constrained_category_stats(constrained_category_stats)

    print(f"\nSaved JSON stats to: {OUTPUT_DIR / 'all_stats.json'}")


if __name__ == "__main__":
    main()
