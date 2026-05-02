import json
from typing import Any, Dict, List, Optional

def build_initial_prompt(task: Dict[str, Any]) -> str:  # CHANGED (task-aware)
    t = task.get("task_type")
    inp = task.get("input", {})
    constraints = task.get("constraints", {})

    if t == "summarization":
        original_text = inp.get("text", "")
        max_words = constraints.get("max_words", 100)
        return (
            f"Summarize the following text in NO MORE THAN {max_words} words.\n"
            f"Return ONLY the summary text. No quotes, no extra commentary.\n\n"
            f"Text:\n{original_text}"
        )

    if t == "code_optimization":
        code = inp.get("code", "")
        lang = inp.get("language", "python")
        goal = constraints.get("goal", "improve readability and performance")
        preserve = constraints.get("preserve_behavior", True)
        avoid_extra = constraints.get("avoid_extra_queries", False)
        return (
            f"You are optimizing {lang} code.\n"
            f"Goal: {goal}.\n"
            f"Constraints:\n"
            f"- Preserve behavior: {preserve}\n"
            f"- Avoid extra queries: {avoid_extra}\n"
            f"Return ONLY the improved code. No explanations.\n\n"
            f"Code:\n{code}"
        )

    if t == "constrained_code_generation":
        lang = inp.get("language", "python")
        prompt = inp.get("prompt", "")
        return (
            f"Write {lang} code that satisfies ALL constraints below.\n"
            f"Return ONLY the code. No explanations.\n\n"
            f"Task:\n{prompt}\n\n"
            f"Constraints:\n{json.dumps(task.get('constraints', {}), ensure_ascii=False)}"
        )

    return "Unsupported task_type."


def build_feedback_prompt(task: Dict[str, Any], current_answer: str, violations: Optional[List[str]] = None) -> str:  # NEW
    t = task.get("task_type")
    inp = task.get("input", {})
    constraints = task.get("constraints", {})

    if t == "summarization":
        original_text = inp.get("text", "")
        max_words = constraints.get("max_words", 100)
        return (
            f"Original text:\n{original_text}\n\n"
            f"Current summary:\n{current_answer}\n\n"
            f"Constraints: <= {max_words} words.\n"
            f"Give short feedback to improve correctness and coverage while staying under the word limit.\n"
            f"Do NOT rewrite the summary—feedback only."
        )

    if t == "code_optimization":
        code = inp.get("code", "")
        lang = inp.get("language", "python")
        return (
            f"Original {lang} code:\n{code}\n\n"
            f"Current improved code:\n{current_answer}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Give SHORT feedback to improve readability/performance while preserving behavior.\n"
            f"Do NOT output code—feedback only."
        )

    if t == "constrained_code_generation":
        # violations-driven feedback so it knows exactly what to fix
        return (
            f"Current code:\n{current_answer}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"These constraints are currently violated:\n- " + "\n- ".join(violations or []) + "\n\n"
            f"Give SHORT feedback describing exactly how to fix the violations.\n"
            f"Do NOT output code—feedback only."
        )

    return "Give feedback."


def build_refine_prompt(task: Dict[str, Any], current_answer: str, feedback: str) -> str:  # NEW
    t = task.get("task_type")
    inp = task.get("input", {})
    constraints = task.get("constraints", {})

    if t == "summarization":
        original_text = inp.get("text", "")
        max_words = constraints.get("max_words", 100)
        return (
            f"Original text:\n{original_text}\n\n"
            f"Current summary:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Rewrite the summary to address the feedback.\n"
            f"Constraints: NO MORE THAN {max_words} words.\n"
            f"Return ONLY the improved summary text."
        )

    if t == "code_optimization":
        code = inp.get("code", "")
        lang = inp.get("language", "python")
        return (
            f"Original {lang} code:\n{code}\n\n"
            f"Current improved code:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Rewrite the improved code to address the feedback while preserving behavior.\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Return ONLY the improved code."
        )

    if t == "constrained_code_generation":
        lang = inp.get("language", "python")
        prompt = inp.get("prompt", "")
        return (
            f"Task:\n{prompt}\n\n"
            f"Current {lang} code:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Rewrite the code so ALL constraints pass.\n"
            f"Return ONLY the code."
        )

    return "Rewrite."
