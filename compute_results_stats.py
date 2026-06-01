import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


SUMMARY_RESULTS_FILE = "summarization_results.json"
CODE_RESULTS_FILE = "code_optimization_results.json"
CONSTRAINED_RESULTS_FILE = "constrained_summary_results.json"

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


def load_json(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)

    if not path.exists():
        print(f"Missing file: {file_path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"No rows to save for {path}")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def fmt(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def collect_similarity_by_round(
    results: List[Dict[str, Any]]
) -> Dict[int, List[float]]:
    values_by_round: Dict[int, List[float]] = {}

    for result in results:
        single_shot_similarity = result.get("single_shot_similarity")

        if isinstance(single_shot_similarity, (int, float)):
            values_by_round.setdefault(0, []).append(float(single_shot_similarity))

        for checkpoint in result.get("checkpoint_results", []):
            round_num = checkpoint.get("round")
            similarity = checkpoint.get("similarity")

            if isinstance(round_num, int) and isinstance(similarity, (int, float)):
                values_by_round.setdefault(round_num, []).append(float(similarity))

    return values_by_round


def compute_similarity_stats(
    results: List[Dict[str, Any]],
    task_name: str,
) -> List[Dict[str, Any]]:
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
                "min_similarity": min(values),
                "max_similarity": max(values),
            }
        )

    return rows


def print_similarity_stats(rows: List[Dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")

    if not rows:
        print("No similarity stats found.")
        return

    print(f"{'Round':<8} {'N':<6} {'Avg Sim':<10} {'Min Sim':<10} {'Max Sim':<10}")

    for row in rows:
        print(
            f"{row['round']:<8} "
            f"{row['num_examples']:<6} "
            f"{fmt(row.get('average_similarity')):<10} "
            f"{fmt(row.get('min_similarity')):<10} "
            f"{fmt(row.get('max_similarity')):<10}"
        )


def collect_constrained_evaluations(
    results: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    evaluations_by_round: Dict[int, List[Dict[str, Any]]] = {}

    for result in results:
        single_eval = result.get("single_shot_constraint_evaluation")

        if isinstance(single_eval, dict) and not single_eval.get("parse_error"):
            evaluations_by_round.setdefault(0, []).append(single_eval)

        for checkpoint in result.get("checkpoint_results", []):
            round_num = checkpoint.get("round")
            evaluation = checkpoint.get("constraint_evaluation")

            if (
                isinstance(round_num, int)
                and isinstance(evaluation, dict)
                and not evaluation.get("parse_error")
            ):
                evaluations_by_round.setdefault(round_num, []).append(evaluation)

    return evaluations_by_round


def get_category_score(
    evaluation: Dict[str, Any],
    category: str,
) -> Optional[float]:
    value = evaluation.get(category)

    if not isinstance(value, dict):
        return None

    score = value.get("score")

    if isinstance(score, (int, float)):
        return float(score)

    return None


def get_category_satisfied(
    evaluation: Dict[str, Any],
    category: str,
) -> Optional[float]:
    value = evaluation.get(category)

    if not isinstance(value, dict):
        return None

    satisfied = value.get("satisfied")

    if isinstance(satisfied, bool):
        return 1.0 if satisfied else 0.0

    return None


def compute_constrained_overall_stats(
    evaluations_by_round: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    rows = []

    for round_num in sorted(evaluations_by_round):
        evaluations = evaluations_by_round[round_num]

        completion_values = []
        overall_pass_values = []
        must_include_counts = []
        must_include_ratios = []
        must_avoid_counts = []
        must_avoid_ratios = []

        for evaluation in evaluations:
            completion = evaluation.get("constraint_completion_ratio")
            if isinstance(completion, (int, float)):
                completion_values.append(float(completion))

            overall_pass = evaluation.get("overall_pass")
            if isinstance(overall_pass, bool):
                overall_pass_values.append(1.0 if overall_pass else 0.0)

            mi_count = evaluation.get("must_include_satisfied_count")
            mi_total = evaluation.get("must_include_total")

            if isinstance(mi_count, (int, float)):
                must_include_counts.append(float(mi_count))

            if (
                isinstance(mi_count, (int, float))
                and isinstance(mi_total, (int, float))
                and mi_total > 0
            ):
                must_include_ratios.append(float(mi_count) / float(mi_total))

            ma_count = evaluation.get("must_avoid_satisfied_count")
            ma_total = evaluation.get("must_avoid_total")

            if isinstance(ma_count, (int, float)):
                must_avoid_counts.append(float(ma_count))

            if (
                isinstance(ma_count, (int, float))
                and isinstance(ma_total, (int, float))
                and ma_total > 0
            ):
                must_avoid_ratios.append(float(ma_count) / float(ma_total))

        rows.append(
            {
                "round": round_num,
                "num_examples": len(evaluations),
                "average_constraint_completion_ratio": safe_mean(completion_values),
                "overall_pass_rate": safe_mean(overall_pass_values),
                "average_must_include_satisfied_count": safe_mean(must_include_counts),
                "average_must_include_ratio": safe_mean(must_include_ratios),
                "average_must_avoid_satisfied_count": safe_mean(must_avoid_counts),
                "average_must_avoid_ratio": safe_mean(must_avoid_ratios),
            }
        )

    return rows


def compute_constrained_category_stats(
    evaluations_by_round: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    rows = []

    for round_num in sorted(evaluations_by_round):
        evaluations = evaluations_by_round[round_num]

        row = {
            "round": round_num,
            "num_examples": len(evaluations),
        }

        for category in GENERAL_EVAL_CATEGORIES:
            scores = []
            pass_values = []

            for evaluation in evaluations:
                score = get_category_score(evaluation, category)
                passed = get_category_satisfied(evaluation, category)

                if score is not None:
                    scores.append(score)

                if passed is not None:
                    pass_values.append(passed)

            row[f"average_{category}_score"] = safe_mean(scores)
            row[f"{category}_pass_rate"] = safe_mean(pass_values)

        rows.append(row)

    return rows


def compute_item_level_stats(
    evaluations_by_round: Dict[int, List[Dict[str, Any]]],
    field_name: str,
    item_type: str,
) -> List[Dict[str, Any]]:
    rows = []

    for round_num in sorted(evaluations_by_round):
        item_values: Dict[str, List[float]] = {}

        for evaluation in evaluations_by_round[round_num]:
            items = evaluation.get(field_name, [])

            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                item_name = item.get("item")
                satisfied = item.get("satisfied")

                if isinstance(item_name, str) and isinstance(satisfied, bool):
                    item_values.setdefault(item_name, []).append(
                        1.0 if satisfied else 0.0
                    )

        for item_name, values in item_values.items():
            rows.append(
                {
                    "round": round_num,
                    "item_type": item_type,
                    "item": item_name,
                    "num_examples": len(values),
                    "satisfaction_rate": safe_mean(values),
                }
            )

    return rows


def print_constrained_overall_stats(rows: List[Dict[str, Any]]) -> None:
    print("\n=== CONSTRAINED SUMMARY OVERALL STATS ===")

    if not rows:
        print("No constrained summary overall stats found.")
        return

    print(
        f"{'Round':<8} "
        f"{'N':<6} "
        f"{'Completion':<12} "
        f"{'Pass Rate':<12} "
        f"{'MustInc Avg':<12} "
        f"{'MustInc Ratio':<14} "
        f"{'MustAvoid Avg':<14} "
        f"{'MustAvoid Ratio':<15}"
    )

    for row in rows:
        print(
            f"{row['round']:<8} "
            f"{row['num_examples']:<6} "
            f"{fmt(row.get('average_constraint_completion_ratio')):<12} "
            f"{fmt(row.get('overall_pass_rate')):<12} "
            f"{fmt(row.get('average_must_include_satisfied_count')):<12} "
            f"{fmt(row.get('average_must_include_ratio')):<14} "
            f"{fmt(row.get('average_must_avoid_satisfied_count')):<14} "
            f"{fmt(row.get('average_must_avoid_ratio')):<15}"
        )


def print_constrained_category_stats(rows: List[Dict[str, Any]]) -> None:
    print("\n=== CONSTRAINED SUMMARY CATEGORY STATS ===")

    if not rows:
        print("No constrained summary category stats found.")
        return

    for row in rows:
        print(f"\nRound {row['round']} | N={row['num_examples']}")

        for category in GENERAL_EVAL_CATEGORIES:
            avg_score = row.get(f"average_{category}_score")
            pass_rate = row.get(f"{category}_pass_rate")

            print(
                f"  {category:<14} "
                f"avg_score={fmt(avg_score)} "
                f"pass_rate={fmt(pass_rate)}"
            )


def print_item_level_stats(rows: List[Dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ITEM-LEVEL STATS ===")

    if not rows:
        print("No item-level stats found.")
        return

    current_round = None

    for row in rows:
        if row["round"] != current_round:
            current_round = row["round"]
            print(f"\nRound {current_round}")
            print(f"{'Item':<40} {'N':<6} {'Satisfaction Rate':<18}")

        print(
            f"{row['item']:<40} "
            f"{row['num_examples']:<6} "
            f"{fmt(row.get('satisfaction_rate')):<18}"
        )


def main() -> None:
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

    constrained_evaluations = collect_constrained_evaluations(constrained_results)

    constrained_overall_stats = compute_constrained_overall_stats(
        constrained_evaluations
    )

    constrained_category_stats = compute_constrained_category_stats(
        constrained_evaluations
    )

    must_include_item_stats = compute_item_level_stats(
        constrained_evaluations,
        field_name="must_include_items",
        item_type="must_include",
    )

    must_avoid_item_stats = compute_item_level_stats(
        constrained_evaluations,
        field_name="must_avoid_items",
        item_type="must_avoid",
    )

    all_stats = {
        "summarization_similarity_stats": summarization_stats,
        "code_optimization_similarity_stats": code_stats,
        "constrained_summary_overall_stats": constrained_overall_stats,
        "constrained_summary_category_stats": constrained_category_stats,
        "must_include_item_stats": must_include_item_stats,
        "must_avoid_item_stats": must_avoid_item_stats,
    }

    write_json(OUTPUT_DIR / "all_stats.json", all_stats)

    write_csv(
        OUTPUT_DIR / "summarization_similarity_stats.csv",
        summarization_stats,
    )

    write_csv(
        OUTPUT_DIR / "code_optimization_similarity_stats.csv",
        code_stats,
    )

    write_csv(
        OUTPUT_DIR / "constrained_summary_overall_stats.csv",
        constrained_overall_stats,
    )

    write_csv(
        OUTPUT_DIR / "constrained_summary_category_stats.csv",
        constrained_category_stats,
    )

    write_csv(
        OUTPUT_DIR / "must_include_item_stats.csv",
        must_include_item_stats,
    )

    write_csv(
        OUTPUT_DIR / "must_avoid_item_stats.csv",
        must_avoid_item_stats,
    )

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

    print_item_level_stats(
        must_include_item_stats,
        title="MUST INCLUDE",
    )

    print_item_level_stats(
        must_avoid_item_stats,
        title="MUST AVOID",
    )

    print(f"\nSaved CSV and JSON stats to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
