import json


# JSON files
def load_input_json(file_path):
    pass


def save_results_json(file_path):
    pass


# Prompts
def build_initial_prompt(task):
    pass


def build_feedback_prompt(task, current_answer):
    pass


def build_refine_prompt(task, current_answer, feedback):
    pass


# LLM call
def call_llm(prompt):
    pass


# Self-improve loop
def self_improve(task) -> dict:
    # returns task in a dictionary format
    pass


def main():
    input_file = "input.json"
    output_file = "results.json"

    tasks = load_input_json(input_file)
    results = []

    for task in tasks:
        self_improve(task)
        results.append(task)

    save_results_json(output_file)


if __name__ == "__main__":
    main()
