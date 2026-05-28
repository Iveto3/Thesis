import json
from typing import Any, Dict, List, Optional

SUMMARY_TASK_TYPES = ("summarization", "constrained_summarization")

def build_initial_prompt(task: Dict[str, Any]) -> str:
    task_type = task.get("task_type")
    task_input = task.get("input", {})
    constraints = task.get("constraints", {})

    if task_type in SUMMARY_TASK_TYPES:
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 150)
        goal = constraints.get("goal", "summarize")

        constraint_text = ""

        if task_type == "constrained_summarization":
            tone = constraints.get("tone")
            style = constraints.get("style")
            must_include = constraints.get("must_include", [])
            must_avoid = constraints.get("must_avoid", [])

            constraint_text = (
                f"\nAdditional constraints:\n"
                f"- Tone: {tone or 'not specified'}\n"
                f"- Style: {style or 'not specified'}\n"
                f"- Must include: {json.dumps(must_include, ensure_ascii=False)}\n"
                f"- Must avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n"
            )

        return (
            f"You are writing a faithful summary.\n\n"
            f"Goal: {goal}\n"
            f"Maximum words: {max_words}\n"
            f"{constraint_text}\n"
            f"Rules:\n"
            f"1. Include only information supported by the source text.\n"
            f"2. Focus on the central facts, actors, numbers, and outcome.\n"
            f"3. Do not add background knowledge or speculation.\n"
            f"4. Preserve names, numbers, places, and dates exactly.\n"
            f"5. Follow all additional constraints exactly.\n"
            f"6. Return only the summary, with no explanation.\n\n"
            f"Source text:\n{text}"
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")
        goal = constraints.get("goal", "optimize readability and performance")

        return (
            f"You are optimizing code.\n\n"
            f"Goal: {goal}\n\n"
            f"Original code:\n{code}\n\n"
            f"Rules:\n"
            f"1. Preserve the exact behavior of the original code.\n"
            f"2. Improve readability and/or performance only when it is safe.\n"
            f"3. Do not add unnecessary abstractions, imports, comments, or features.\n"
            f"4. Do not wrap the answer in markdown code fences.\n"
            f"5. Return only the improved code."
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

    if task_type in SUMMARY_TASK_TYPES:
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 150)
        goal = constraints.get("goal", "summarize")

        constraint_text = ""

        if task_type == "constrained_summarization":
            tone = constraints.get("tone")
            style = constraints.get("style")
            must_include = constraints.get("must_include", [])
            must_avoid = constraints.get("must_avoid", [])

            constraint_text = (
                f"\nTask constraints:\n"
                f"- Goal: {goal}\n"
                f"- Maximum words: {max_words}\n"
                f"- Tone: {tone or 'not specified'}\n"
                f"- Style: {style or 'not specified'}\n"
                f"- Must include: {json.dumps(must_include, ensure_ascii=False)}\n"
                f"- Must avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n"
            )

        return (
            f"You are evaluating a summary. Your job is to give precise, "
            f"actionable feedback — not to rewrite the summary.\n\n"
            f"Source text:\n{text}\n\n"
            f"Current summary:\n{current_answer}\n"
            f"{constraint_text}\n"
            f"Evaluate the summary on these dimensions, in order:\n"
            f"1. FAITHFULNESS — Are all claims supported by the source? "
            f"Flag unsupported, exaggerated, or hallucinated claims.\n"
            f"2. COVERAGE — Are the most important facts included? "
            f"Focus on who, what, where, main outcome, and key numbers.\n"
            f"3. ENTITIES — Are names, numbers, places, and dates reproduced "
            f"exactly as in the source?\n"
            f"4. LENGTH — Is the summary under {max_words} words?\n"
            f"5. CONSTRAINTS — Does it follow the requested goal, tone, style, "
            f"must_include, and must_avoid constraints?\n\n"
            f"Output format, using exactly these labels:\n"
            f"FAITHFULNESS: <one issue, or 'OK'>\n"
            f"COVERAGE: <one issue, or 'OK'>\n"
            f"ENTITIES: <one issue, or 'OK'>\n"
            f"LENGTH: <word count and 'OK' or 'OVER'>\n"
            f"CONSTRAINTS: <one issue, or 'OK'>\n"
            f"VERDICT: <NEEDS_REVISION or NO_CHANGES_NEEDED>\n\n"
            f"If all dimensions are OK, VERDICT must be NO_CHANGES_NEEDED. "
            f"Do not invent issues. Do not comment on general style unless it violates "
            f"an explicit constraint. Do not rewrite the summary."
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")

        return (
            f"You are evaluating an attempted code optimization. "
            f"Give precise feedback only; do not output code.\n\n"
            f"Original code:\n{code}\n\n"
            f"Current optimized code:\n{current_answer}\n\n"
            f"Evaluate the current optimized code on these dimensions:\n"
            f"1. BEHAVIOR — Does it preserve the original behavior exactly?\n"
            f"2. SIMPLIFICATION — Does it actually simplify or improve the code?\n"
            f"3. READABILITY — Is the result clearer without becoming more complex?\n"
            f"4. MINIMALITY — Did it avoid unnecessary imports, comments, helpers, "
            f"or extra logic?\n\n"
            f"Output format, using exactly these labels:\n"
            f"BEHAVIOR: <one issue, or 'OK'>\n"
            f"SIMPLIFICATION: <one issue, or 'OK'>\n"
            f"READABILITY: <one issue, or 'OK'>\n"
            f"MINIMALITY: <one issue, or 'OK'>\n"
            f"VERDICT: <NEEDS_REVISION or NO_CHANGES_NEEDED>\n\n"
            f"If the current code is already good enough, use NO_CHANGES_NEEDED. "
            f"Do not rewrite the code."
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

    if task_type in SUMMARY_TASK_TYPES:
        text = task_input.get("text", "")
        max_words = constraints.get("max_words", 150)
        goal = constraints.get("goal", "summarize")

        constraint_text = ""

        if task_type == "constrained_summarization":
            tone = constraints.get("tone")
            style = constraints.get("style")
            must_include = constraints.get("must_include", [])
            must_avoid = constraints.get("must_avoid", [])

            constraint_text = (
                f"\nTask constraints:\n"
                f"- Goal: {goal}\n"
                f"- Maximum words: {max_words}\n"
                f"- Tone: {tone or 'not specified'}\n"
                f"- Style: {style or 'not specified'}\n"
                f"- Must include: {json.dumps(must_include, ensure_ascii=False)}\n"
                f"- Must avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n"
            )

        return (
            f"Source text:\n{text}\n\n"
            f"Current summary:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n"
            f"{constraint_text}\n"
            f"Rewrite the summary to address ONLY the issues flagged in the feedback.\n\n"
            f"Rules:\n"
            f"1. For any dimension marked 'OK', keep the relevant wording from the "
            f"current summary unchanged. Do not paraphrase for style.\n"
            f"2. Add, remove, or correct content only if the feedback requires it.\n"
            f"3. Preserve named entities, numbers, places, and dates exactly as in the source.\n"
            f"4. Follow all task constraints exactly.\n"
            f"5. Stay under {max_words} words.\n"
            f"6. If the feedback verdict is NO_CHANGES_NEEDED, return the current summary unchanged.\n\n"
            f"Return ONLY the revised summary, no explanation."
        )

    if task_type == "code_optimization":
        code = task_input.get("code", "")

        return (
            f"Original code:\n{code}\n\n"
            f"Current optimized code:\n{current_answer}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Rewrite the code to address ONLY the issues flagged in the feedback.\n\n"
            f"Rules:\n"
            f"1. Preserve the original behavior exactly.\n"
            f"2. Keep the solution minimal.\n"
            f"3. Do not add imports, comments, helper functions, or extra logic unless required.\n"
            f"4. If the feedback verdict is NO_CHANGES_NEEDED, return the current code unchanged.\n"
            f"5. Do not wrap the answer in markdown code fences.\n\n"
            f"Return ONLY the revised code."
        )

    return "Rewrite."

def build_constraint_evaluation_prompt(
    task: Dict[str, Any],
    candidate_answer: str,
) -> str:
    task_input = task.get("input", {})
    constraints = task.get("constraints", {})

    text = task_input.get("text", "")
    max_words = constraints.get("max_words", 150)
    goal = constraints.get("goal", "summarize")
    tone = constraints.get("tone", "not specified")
    style = constraints.get("style", "not specified")
    must_include = constraints.get("must_include", [])
    must_avoid = constraints.get("must_avoid", [])

    return (
        f"You are an independent evaluator of a constrained news summary.\n"
        f"Your job is to judge whether the candidate summary satisfies the task "
        f"and constraints. Do not rewrite the summary.\n\n"
        f"Source article:\n{text}\n\n"
        f"Candidate summary:\n{candidate_answer}\n\n"
        f"Task constraints:\n"
        f"- Goal: {goal}\n"
        f"- Maximum words: {max_words}\n"
        f"- Tone: {tone}\n"
        f"- Style: {style}\n"
        f"- Must include: {json.dumps(must_include, ensure_ascii=False)}\n"
        f"- Must avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n\n"
        f"Evaluate each dimension using this scale:\n"
        f"0 = not satisfied\n"
        f"1 = partially satisfied\n"
        f"2 = fully satisfied\n\n"
        f"Dimensions:\n"
        f"1. faithfulness: Are all claims supported by the source article?\n"
        f"2. coverage: Does it include the main event, key people or organizations, "
        f"and main outcome?\n"
        f"3. tone: Does it sound like a news reporter?\n"
        f"4. style: Does it follow the requested rhyming style while remaining factual?\n"
        f"5. must_include: Does it include the required information?\n"
        f"6. must_avoid: Does it avoid unsupported claims, extra details, and jokes?\n"
        f"7. length: Is it under {max_words} words?\n"
        f"8. entities: Are names, numbers, places, and dates accurate?\n\n"
        f"Return ONLY valid JSON in this exact structure:\n"
        f"{{\n"
        f'  "faithfulness": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "coverage": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "tone": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "style": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "must_include": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "must_avoid": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "length": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "entities": {{"score": 0, "pass": false, "issue": ""}},\n'
        f'  "total_score": 0,\n'
        f'  "overall_pass": false,\n'
        f'  "brief_reason": ""\n'
        f"}}"
    )
