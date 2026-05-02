import json
from typing import Any, Dict, List, Optional


def build_initial_prompt(task: Dict[str, Any]) -> str:
    task_type = task.get("task_type")
    task_input = task.get("input", {})
    constraints = task.get("constraints", {})

    if task_type == "summarization":
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 150)
        goal = constraints.get("goal", "summarize")

        return (
            f"Task: {goal}.\n"
            f"Maximum words: {max_words}.\n"
            f"Return only the summary of the text with no unecessary details.\n\n"
            f"Text:\n{text}"
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")
        # language = task_input.get("language", "python")
        goal = constraints.get("goal", "optimize readability and performance")

        return (
            # f"Optimize this {language} code.\n"
            f"Goal: {goal}.\n"
            f"Rules:\n"
            f"1. Preserve the original behavior of the code.\n"
            f"2. Do not add extra complicated code.\n"
            f"3. Return only the improved code with no explanations.\n\n"
            f"Code:\n{code}"
        )

    if task_type == "constrained_code_generation":
        language = task_input.get("language", "python")
        prompt = task_input.get("prompt", "")

        return (
            f"Write {language} code for this task:\n"
            f"{prompt}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Return only the code and no explanations."
        )

    return "Unsupported task_type."


def build_feedback_prompt(
    task: Dict[str, Any],
    current_answer: str,
    violations: Optional[List[str]] = None,
) -> str:
    task_type = task.get("task_type")
    task_input = task.get("input", {})
    constraints = task.get("constraints", {})

    if task_type == "summarization":
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 150)

        return (
            f"Original text:\n{text}\n\n"
            f"Current summary:\n{current_answer}\n\n"
            f"Give short feedback to improve the summary.\n"
            f"It must stay under {max_words} words.\n"
            f"Do NOT rewrite the summary. Feedback only."
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")
        # language = task_input.get("language", "python")

        return (
            # f"Original {language} code:\n{code}\n\n"
            f"Original code:\n{code}\n\n"
            f"Current improved code:\n{current_answer}\n\n"
            f"Give short feedback to improve readability/performance.\n"
            f"The code must preserve behavior and avoid extra elements added.\n"
            f"Do NOT output code. Feedback only."
        )

    if task_type == "constrained_code_generation":
        violation_text = "\n- ".join(violations or [])

        return (
            f"Current code:\n{current_answer}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Violated constraints:\n- {violation_text}\n\n"
            f"Give short feedback explaining how to fix the violations.\n"
            f"Do NOT output code. Feedback only."
        )

    return "Give feedback."


def build_refine_prompt(
    task: Dict[str, Any],
    current_answer: str,
    feedback: str,
) -> str:
    task_type = task.get("task_type")
    task_input = task.get("input", {})
    constraints = task.get("constraints", {})

    if task_type == "summarization":
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 100)

        return (
            f"Original text:\n{text}\n\n"
            f"Current summary:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Rewrite the summary using the feedback.\n"
            f"Maximum words: {max_words}.\n"
            f"Return ONLY the improved summary."
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")
        # language = task_input.get("language", "python")

        return (
            f"Original code:\n{code}\n\n"
            f"Current improved code:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Rewrite the code using the feedback.\n"
            f"Rules:\n"
            f"1. Preserve the original behavior.\n"
            f"2. Do not add extra code elements.\n"
            f"3. Return ONLY the improved code."
        )

    if task_type == "constrained_code_generation":
        language = task_input.get("language", "python")
        prompt = task_input.get("prompt", "")

        return (
            f"Task:\n{prompt}\n\n"
            f"Current {language} code:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"Rewrite the code so all constraints pass.\n"
            f"Return ONLY the code."
        )

    return "Rewrite."
