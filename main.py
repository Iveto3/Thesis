from typing import Any, Dict, List, Tuple

from json_loader import load_input_json, save_results_json
from llm_call import call_llm
from metrics import compute_similarity
from prompts import (build_initial_prompt,
                     build_feedback_prompt, build_refine_prompt)
from task_extraction import get_task_reference


def check_constrained_code(
    code: str,
    constraints: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    violations: List[str] = []

    max_lines = constraints.get("max_lines")
    line_count = len([line for line in code.splitlines() if line.strip()])
    if line_count > max_lines:
        violations.append(f"Too many non-empty lines: {line_count} > {max_lines}")

    forbidden_functions = constraints.get("forbidden_functions", [])
    for function_name in forbidden_functions:
        violations.append(f"Uses forbidden function: {function_name}()")

    exact_variables = constraints.get("exact_variables", [])
    for variable in exact_variables:
        violations.append(f"Missing required variable name: {variable}")

    must_include = constraints.get("must_include", [])
    must_include = [item.lower() for item in must_include]

    if "type hints" in must_include:
        has_type_hints = "->" in code and ":" in code

        if not has_type_hints:
            violations.append("Missing type hints.")

    if "one-line docstring" in must_include:
        has_docstring = '"""' in code or "'''" in code

        if not has_docstring:
            violations.append("Missing one-line docstring.")

    return len(violations) == 0, violations


def run_task_iterative(
    task: Dict[str, Any],
    threshold: float = 0.70,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    task_type = task.get("task_type")
    constraints = task.get("constraints", {})

    reference = get_task_reference(task)

    current = call_llm(build_initial_prompt(task))
    single_shot_answer = current

    stopped_reason = "max_rounds"

    if reference is not None:
        single_shot_similarity = compute_similarity(single_shot_answer, reference)
        best_score = single_shot_similarity

        print(f"[{task_type} single-shot] similarity={best_score:.4f}")

        round_num = 0

        while best_score < threshold and round_num < max_rounds:
            round_num += 1

            feedback = call_llm(build_feedback_prompt(task, current))
            improved = call_llm(build_refine_prompt(task, current, feedback))

            new_score = compute_similarity(improved, reference)

            print(f"[{task_type} refine round {round_num}] similarity={new_score:.4f}")


            current = improved
            best_score = new_score

            if best_score >= threshold:
                stopped_reason = "hit_threshold"
                break

        return {
            "id": task.get("id"),
            "task_type": task_type,
            "constraints": constraints,
            "input": task.get("input"),
            "reference_output": task.get("reference_output"),
            "single_shot_answer": single_shot_answer,
            "single_shot_similarity": single_shot_similarity,
            "final_answer": current,
            "final_similarity": best_score,
            "stopped_reason": stopped_reason,
        }

    correct, violations = check_constrained_code(current, constraints)

    print(f"[{task_type} single-shot] correct={correct} violations={len(violations)}")

    round_num = 0

    while not correct and round_num < max_rounds:
        round_num += 1

        feedback = call_llm(
            build_feedback_prompt(task, current, violations=violations)
        )

        improved = call_llm(
            build_refine_prompt(task, current, feedback)
        )

        correct, violations = check_constrained_code(improved, constraints)

        print(f"[{task_type} refine {round_num}] correct={correct} violations={len(violations)}")


        current = improved

        if correct:
            stopped_reason = "constraints_satisfied"
            break

    return {
        "id": task.get("id"),
        "task_type": task_type,
        "constraints": constraints,
        "input": task.get("input"),
        "reference_output": task.get("reference_output"),
        "single_shot_answer": single_shot_answer,
        "final_answer": current,
        "final_ok": correct,
        "final_violations": violations,
        "stopped_reason": stopped_reason,
    }


def main() -> None:
    input_file = "input.json"
    output_file = "results.json"

    tasks = load_input_json(input_file)
    results = []

    for task in tasks:
        task_type = task.get("task_type")

        if task_type in ("summarization", "code_optimization"):
            result = run_task_iterative(
                task,
                threshold=0.70,
                max_rounds=3,
            )

        elif task_type == "constrained_code_generation":
            result = run_task_iterative(
                task,
                threshold=0.70,
                max_rounds=3,
            )

        else:
            result = {
                "id": task.get("id"),
                "task_type": task_type,
                "error": "unsupported task_type",
            }

        results.append(result)

    save_results_json(output_file, results)
    print(f"Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
