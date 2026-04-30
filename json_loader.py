import json


def load_input_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_json(file_path, results):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
