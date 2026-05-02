from typing import Any, Dict, Optional


def get_task_input(task: Dict[str, Any]) -> str:
    task_type = task.get("task_type")
    inp = task.get("input", {})

    if task_type == "summarization":
        return inp.get("text", "")
    if task_type == "code_optimization":
        return inp.get("code", "")
    if task_type == "constrained_code_generation":
        return inp.get("prompt", "")

    return ""


def get_task_reference(task: Dict[str, Any]) -> Optional[str]:
    task_type = task.get("task_type")
    ref = task.get("reference_output")

    if not ref:
        return None
    if task_type == "summarization":
        return ref.get("summary")
    if task_type == "code_optimization":
        return ref.get("optimized_code")

    return None
