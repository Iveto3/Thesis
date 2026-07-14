import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from compute_results_stats import main as compute_stats_main
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
    """ Parses JSON from LLM output. """
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
    """ Evaluates constraints. """
    raw_eval = call_llm(
        build_constraint_evaluation_prompt(task, answer),
        model=judge_model,
    )
    return parse_json_from_llm(raw_eval)


def load_existing_results(output_file: str) -> Dict[str, Dict[str, Any]]:
    """ Loads existing results. """
    path = Path(output_file)

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {}

        results_by_id = {}

        for result in data:
            if not isinstance(result, dict):
                continue

            task_id = result.get("id")

            if isinstance(task_id, str):
                results_by_id[task_id] = result

        return results_by_id

    except Exception as error:
        print(
            f"Could not load existing results from {output_file}: {error}",
            flush=True,
        )
        return {}


def save_results_in_task_order(
    output_file: str,
    tasks: List[Dict[str, Any]],
    results_by_id: Dict[str, Dict[str, Any]],
) -> None:
    """ Saves results in ordered list of tasks. """
    ordered_results = []

    for task in tasks:
        task_id = task.get("id")

        if isinstance(task_id, str) and task_id in results_by_id:
            ordered_results.append(results_by_id[task_id])

    extra_results = [
        result
        for task_id, result in results_by_id.items()
        if task_id not in {task.get("id") for task in tasks}
    ]

    ordered_results.extend(extra_results)
    save_results_json(output_file, ordered_results)


def add_or_replace_checkpoint(
    checkpoint_results: List[Dict[str, Any]],
    checkpoint: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """ Adds or replaces checkpoints. """
    round_num = checkpoint.get("round")
    updated = [
        existing
        for existing in checkpoint_results
        if existing.get("round") != round_num
    ]
    updated.append(checkpoint)
    updated.sort(key=lambda item: item.get("round", -1))
    return updated


def is_completed_result(
    result: Optional[Dict[str, Any]],
    rounds: int,
) -> bool:
    """ Checks if result is already completed for given rounds. """
    if not isinstance(result, dict):
        return False

    if result.get("stopped_reason") != "fixed_rounds_completed":
        return False

    final_round = result.get("final_round")
    last_completed_round = result.get("last_completed_round")

    return final_round == rounds or last_completed_round == rounds


def run_task_iterative(
    task: Dict[str, Any],
    rounds: int = 50,
    generation_model: str = "gemma3:4b",
    judge_model: str = "gemma3:4b",
    existing_result: Optional[Dict[str, Any]] = None,
    save_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """ Runs a task iteratively. """
    task_id = task.get("id", "unknown_id")
    task_type = task.get("task_type")
    constraints = task.get("constraints", {})
    reference = get_task_reference(task)

    existing_result = existing_result or {}
    checkpoint_results = list(existing_result.get("checkpoint_results", []))

    def save_progress(result: Dict[str, Any]) -> None:
        """ Saves progress of task. """
        if save_callback is not None:
            save_callback(result)

    if existing_result.get("single_shot_answer"):
        single_shot_answer = existing_result["single_shot_answer"]
        current = (
            existing_result.get("current_answer")
            or existing_result.get("final_answer")
            or single_shot_answer
        )
        last_completed_round = int(existing_result.get("last_completed_round", 0))

        print(
            f"[{task_type} | {task_id}] resuming from round {last_completed_round}",
            flush=True,
        )

    else:
        current = call_llm(
            build_initial_prompt(task),
            model=generation_model,
        )

        single_shot_answer = current
        last_completed_round = 0

    if reference is not None:
        single_shot_similarity = existing_result.get("single_shot_similarity")

        if not isinstance(single_shot_similarity, (int, float)):
            single_shot_similarity = compute_similarity(
                single_shot_answer, reference)

            print(
                f"[{task_type} | {task_id} | single-shot] "
                f"similarity={single_shot_similarity:.4f}",
                flush=True,
            )

        best_answer = existing_result.get("best_answer", single_shot_answer)
        best_similarity = existing_result.get("best_similarity", single_shot_similarity)
        best_round = existing_result.get("best_round", 0)

        current_similarity = existing_result.get("current_similarity")

        if not isinstance(current_similarity, (int, float)):
            current_similarity = single_shot_similarity

        result = {
            "id": task_id,
            "task_type": task_type,
            "constraints": constraints,
            "single_shot_answer": single_shot_answer,
            "single_shot_similarity": float(single_shot_similarity),
            "last_completed_round": last_completed_round,
            "current_answer": current,
            "current_similarity": float(current_similarity),
            "final_round": last_completed_round,
            "final_answer": current,
            "final_similarity": float(current_similarity),
            "best_round": best_round,
            "best_answer": best_answer,
            "best_similarity": float(best_similarity),
            "checkpoint_results": checkpoint_results,
            "stopped_reason": "in_progress",
        }

        save_progress(result)

        for round_num in range(last_completed_round + 1, rounds + 1):
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
                f"[{task_type} | {task_id} | round {round_num}] "
                f"similarity={similarity:.4f}",
                flush=True,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = current
                best_round = round_num

            if round_num in CHECKPOINT_ROUNDS:
                checkpoint_results = add_or_replace_checkpoint(
                    checkpoint_results,
                    {
                        "round": round_num,
                        "answer": current,
                        "similarity": similarity,
                    },
                )

            result.update(
                {
                    "last_completed_round": round_num,
                    "current_answer": current,
                    "current_similarity": similarity,
                    "final_round": round_num,
                    "final_answer": current,
                    "final_similarity": similarity,
                    "best_round": best_round,
                    "best_answer": best_answer,
                    "best_similarity": best_similarity,
                    "checkpoint_results": checkpoint_results,
                    "stopped_reason": "in_progress",
                }
            )

            save_progress(result)

        result["stopped_reason"] = "fixed_rounds_completed"
        result["final_round"] = rounds
        save_progress(result)
        return result

    if task_type == "constrained_summarization":
        single_shot_constraint_evaluation = existing_result.get(
            "single_shot_constraint_evaluation"
        )

        if not isinstance(single_shot_constraint_evaluation, dict):
            print(
                f"[{task_type} | {task_id} | single-shot] generated initial answer",
                flush=True,
            )

            single_shot_constraint_evaluation = evaluate_constraints(
                task,
                single_shot_answer,
                judge_model=judge_model,
            )

        result = {
            "id": task_id,
            "task_type": task_type,
            "constraints": constraints,
            "single_shot_answer": single_shot_answer,
            "single_shot_constraint_evaluation": single_shot_constraint_evaluation,
            "last_completed_round": last_completed_round,
            "current_answer": current,
            "final_round": last_completed_round,
            "final_answer": current,
            "checkpoint_results": checkpoint_results,
            "stopped_reason": "in_progress",
        }

        save_progress(result)

        for round_num in range(last_completed_round + 1, rounds + 1):
            feedback = call_llm(
                build_feedback_prompt(task, current),
                model=generation_model,
            )

            current = call_llm(
                build_refine_prompt(task, current, feedback),
                model=generation_model,
            )

            print(
                f"[{task_type} | {task_id} | round {round_num}] completed",
                flush=True,
            )

            if round_num in CHECKPOINT_ROUNDS:
                constraint_evaluation = evaluate_constraints(
                    task,
                    current,
                    judge_model=judge_model,
                )

                checkpoint_results = add_or_replace_checkpoint(
                    checkpoint_results,
                    {
                        "round": round_num,
                        "answer": current,
                        "constraint_evaluation": constraint_evaluation,
                    },
                )

            result.update(
                {
                    "last_completed_round": round_num,
                    "current_answer": current,
                    "final_round": round_num,
                    "final_answer": current,
                    "checkpoint_results": checkpoint_results,
                    "stopped_reason": "in_progress",
                }
            )

            save_progress(result)

        result["stopped_reason"] = "fixed_rounds_completed"
        result["final_round"] = rounds
        save_progress(result)
        return result

    return {
        "id": task_id,
        "task_type": task_type,
        "constraints": constraints,
        "single_shot_answer": single_shot_answer,
        "last_completed_round": last_completed_round,
        "current_answer": current,
        "error": "No reference output available for this task type.",
        "stopped_reason": "error",
    }


def run_experiment(
    input_file: str,
    output_file: str,
    rounds: int,
    generation_model: str,
    judge_model: str,
) -> bool:
    """ Runs the experiment. """
    tasks = load_input_json(input_file)
    existing_results_by_id = load_existing_results(output_file)
    results_by_id = dict(existing_results_by_id)

    print("\nRunning experiment:", flush=True)
    print(f"Input: {input_file}", flush=True)
    print(f"Output: {output_file}", flush=True)
    print(f"Rounds: {rounds}", flush=True)
    print(f"Already have {len(existing_results_by_id)} saved results\n", flush=True)

    for index, task in enumerate(tasks, start=1):
        task_id = task.get("id", "unknown_id")
        task_type = task.get("task_type")
        existing_result = results_by_id.get(task_id)

        if is_completed_result(existing_result, rounds):
            print(
                f"[{task_type} | {task_id}] skipped "
                f"({index}/{len(tasks)}) already completed",
                flush=True,
            )
            continue

        def save_callback(result: Dict[str, Any]) -> None:
            results_by_id[task_id] = result
            save_results_in_task_order(output_file, tasks, results_by_id)

        try:
            if task_type in SUPPORTED_TASK_TYPES:
                result = run_task_iterative(
                    task,
                    rounds=rounds,
                    generation_model=generation_model,
                    judge_model=judge_model,
                    existing_result=existing_result,
                    save_callback=save_callback,
                )
            else:
                result = {
                    "id": task_id,
                    "task_type": task_type,
                    "error": "unsupported task_type",
                    "stopped_reason": "error",
                }

            results_by_id[task_id] = result
            save_results_in_task_order(output_file, tasks, results_by_id)

            print(
                f"[{task_type} | {task_id}] saved progress "
                f"({index}/{len(tasks)}) to {output_file}",
                flush=True,
            )

        except Exception as error:
            error_result = {
                "id": task_id,
                "task_type": task_type,
                "error": str(error),
                "stopped_reason": "exception",
            }

            if isinstance(existing_result, dict):
                error_result.update(existing_result)
                error_result["error"] = str(error)
                error_result["stopped_reason"] = "exception"

            results_by_id[task_id] = error_result
            save_results_in_task_order(output_file, tasks, results_by_id)

            print(
                f"[{task_type} | {task_id}] ERROR saved to {output_file}: {error}",
                flush=True,
            )

            raise

    save_results_in_task_order(output_file, tasks, results_by_id)

    completed_count = sum(
        1
        for task in tasks
        if is_completed_result(results_by_id.get(task.get("id")), rounds)
    )

    print(
        f"Saved {len(results_by_id)} results to {output_file}. "
        f"Completed {completed_count}/{len(tasks)} tasks.",
        flush=True,
    )

    return completed_count == len(tasks)


def main() -> None:
    """The main function."""
    rounds = 50
    generation_model = "gemma3:4b"
    judge_model = "gemma3:4b"

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    experiments = [
        {
            "input_file": "input/summarization_dataset_input.json",
            "output_file": results_dir / "summarization_results.json",
        },
        {
            "input_file": "input/constrained_summary_input.json",
            "output_file": results_dir / "constrained_summary_results.json",
        },
        {
            "input_file": "input/code_optimization_input.json",
            "output_file": results_dir / "code_optimization_results.json",
        },
    ]

    all_experiments_completed = True

    for experiment in experiments:
        experiment_completed = run_experiment(
            input_file=experiment["input_file"],
            output_file=experiment["output_file"],
            rounds=rounds,
            generation_model=generation_model,
            judge_model=judge_model,
        )

        if not experiment_completed:
            all_experiments_completed = False

    if all_experiments_completed:
        print("\nAll experiments finished. Computing result statistics...",
              flush=True)
        compute_stats_main()
    else:
        print(
            "\nSome experiments are incomplete. "
            "Run main.py again to resume from the saved checkpoints.",
            flush=True,
        )


if __name__ == "__main__":
    main()
