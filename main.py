import json
from typing import Any, Dict

from json_loader import load_input_json, save_results_json
from llm_call import call_llm
from metrics import compute_similarity
from prompts import (
    build_initial_prompt,
    build_feedback_prompt,
    build_refine_prompt,
    build_constraint_evaluation_prompt,
)
from task_extraction import get_task_reference


SUPPORTED_TASK_TYPES = (
    "summarization",
    "constrained_summarization",
    "code_optimization",
)

CHECKPOINT_ROUNDS = {1, 3, 5, 10, 50}


def parse_json_from_llm(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "parse_error": True,
        "raw_evaluation": raw_text,
    }


def evaluate_constraints(
    task: Dict[str, Any],
    answer: str,
    judge_model: str,
) -> Dict[str, Any]:
    raw_eval = call_llm(
        build_constraint_evaluation_prompt(task, answer),
        model=judge_model,
    )
    return parse_json_from_llm(raw_eval)


def run_task_iterative(
    task: Dict[str, Any],
    rounds: int = 50,
    generation_model: str = "gemma3:4b",
    judge_model: str = "gemma3:4b",
) -> Dict[str, Any]:
    task_type = task.get("task_type")
    constraints = task.get("constraints", {})
    reference = get_task_reference(task)

    current = call_llm(
        build_initial_prompt(task),
        model=generation_model,
    )

    single_shot_answer = current
    checkpoint_results = []

    if reference is not None:
        single_shot_similarity = compute_similarity(single_shot_answer, reference)

        print(f"[{task_type} single-shot] similarity={single_shot_similarity:.4f}")

        best_answer = single_shot_answer
        best_similarity = single_shot_similarity
        best_round = 0

        final_answer = single_shot_answer
        final_similarity = single_shot_similarity

        for round_num in range(1, rounds + 1):
            feedback = call_llm(
                build_feedback_prompt(task, current),
                model=generation_model,
            )

            current = call_llm(
                build_refine_prompt(task, current, feedback),
                model=generation_model,
            )

            similarity = compute_similarity(current, reference)

            print(
                f"[{task_type} round {round_num}] "
                f"similarity={similarity:.4f}"
            )

            final_answer = current
            final_similarity = similarity

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = current
                best_round = round_num

            if round_num in CHECKPOINT_ROUNDS:
                checkpoint_results.append(
                    {
                        "round": round_num,
                        "answer": current,
                        "similarity": similarity,
                    }
                )

        return {
            "id": task.get("id"),
            "task_type": task_type,
            "constraints": constraints,
            "single_shot_answer": single_shot_answer,
            "single_shot_similarity": single_shot_similarity,
            "final_round": rounds,
            "final_answer": final_answer,
            "final_similarity": final_similarity,
            "best_round": best_round,
            "best_answer": best_answer,
            "best_similarity": best_similarity,
            "checkpoint_results": checkpoint_results,
            "stopped_reason": "fixed_rounds_completed",
        }

    if task_type == "constrained_summarization":
        print(f"[{task_type} single-shot] generated initial answer")

        single_shot_constraint_evaluation = evaluate_constraints(
            task,
            single_shot_answer,
            judge_model=judge_model,
        )

        final_answer = single_shot_answer

        for round_num in range(1, rounds + 1):
            feedback = call_llm(
                build_feedback_prompt(task, current),
                model=generation_model,
            )

            current = call_llm(
                build_refine_prompt(task, current, feedback),
                model=generation_model,
            )

            final_answer = current

            print(f"[{task_type} round {round_num}] completed")

            if round_num in CHECKPOINT_ROUNDS:
                constraint_evaluation = evaluate_constraints(
                    task,
                    current,
                    judge_model=judge_model,
                )

                checkpoint_results.append(
                    {
                        "round": round_num,
                        "answer": current,
                        "constraint_evaluation": constraint_evaluation,
                    }
                )

        return {
            "id": task.get("id"),
            "task_type": task_type,
            "constraints": constraints,
            "single_shot_answer": single_shot_answer,
            "single_shot_constraint_evaluation": single_shot_constraint_evaluation,
            "final_round": rounds,
            "final_answer": final_answer,
            "checkpoint_results": checkpoint_results,
            "stopped_reason": "fixed_rounds_completed",
        }

    return {
        "id": task.get("id"),
        "task_type": task_type,
        "constraints": constraints,
        "single_shot_answer": single_shot_answer,
        "error": "No reference output available for this task type.",
        "stopped_reason": "error",
    }


def run_experiment(
    input_file: str,
    output_file: str,
    rounds: int,
    generation_model: str,
    judge_model: str,
) -> None:
    tasks = load_input_json(input_file)
    results = []

    print(f"\nRunning experiment:")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Rounds: {rounds}\n")

    for task in tasks:
        task_type = task.get("task_type")

        if task_type in SUPPORTED_TASK_TYPES:
            result = run_task_iterative(
                task,
                rounds=rounds,
                generation_model=generation_model,
                judge_model=judge_model,
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


def main() -> None:
    rounds = 50
    generation_model = "gemma3:4b"
    judge_model = "gemma3:4b"

    experiments = [
        {
            "input_file": "summarization_dataset_input.json",
            "output_file": "summarization_results_50.json",
            "enabled": True,
        },
        {
            "input_file": "constrained_summary_input.json",
            "output_file": "constrained_summary_results_50.json",
            "enabled": True,
        },
        {
            "input_file": "code_optimization_input.json",
            "output_file": "code_optimization_results_50.json",
            "enabled": False,
        },
    ]

    for experiment in experiments:
        if not experiment["enabled"]:
            continue

        run_experiment(
            input_file=experiment["input_file"],
            output_file=experiment["output_file"],
            rounds=rounds,
            generation_model=generation_model,
            judge_model=judge_model,
        )


if __name__ == "__main__":
    main()
