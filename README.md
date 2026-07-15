# Iterative LLM Self-Refinement Evaluation

This repository compares **single-pass generation** with an **iterative self-refinement workflow** across three tasks:

- summarization;
- constrained summarization;
- code optimization.

The experiments use `gemma3:4b` through Ollama. The same model generates the initial response, evaluates it, provides feedback, and revises it. For constrained summarization, the same model is also used as the LLM judge.

## Workflow

For each example, the pipeline:

1. generates a single-pass response;
2. asks the model to evaluate the current response;
3. asks the model to revise it using the feedback;
4. repeats the feedback and revision loop for 50 rounds;
5. saves results at the single-pass and refinement rounds 1, 3, 5, 10, and 50.

Progress is saved after every round. If an experiment is interrupted, running `main.py` again resumes it from the latest saved round.

## Evaluation

### Summarization

Summarization is evaluated using:

- cosine similarity with `sentence-transformers/all-MiniLM-L6-v2`;
- ROUGE-Lsum F1;
- BERTScore F1 with `roberta-large`.

### Constrained summarization

An LLM judge evaluates:

- goal satisfaction;
- faithfulness;
- coverage;
- tone;
- style;
- entity accuracy;
- must-include requirements;
- must-avoid requirements.

Length is checked separately using a fixed word-count rule.

The project computes:

- category-specific pass rates;
- average constraint completion ratio;
- overall pass rate.

### Code optimization

Code optimization is evaluated using cosine similarity to the reference optimized code.

The generated code is not executed, unit-tested, or benchmarked.

## Requirements

The project requires:

- Python;
- Ollama;
- the `gemma3:4b` model.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the model:

```bash
ollama pull gemma3:4b
```

Make sure Ollama is running before starting the experiment.


## Execution Order

Run the project in the following order.

### 1. Prepare the input files

Run the dataset-loading scripts in the `loading_datasets_scripts/` directory.

These scripts must create the following files:

```text
input/
├── summarization_dataset_input.json
├── constrained_summary_input.json
└── code_optimization_input.json
```

All three input files must exist before running `main.py`.

### 2. Run the experiments

```bash
python main.py
```

`main.py` runs the following experiments in sequence:

1. summarization;
2. constrained summarization;
3. code optimization.

Each experiment uses 50 refinement rounds and saves checkpoints at rounds 0, 1, 3, 5, 10, and 50.

When all experiments are complete, `main.py` automatically runs:

```bash
python compute_results_stats.py
```

This creates the initial aggregate statistics in:

```text
computed_stats/all_stats.json
```

You do not need to run `compute_results_stats.py` manually after a successful full run of `main.py`.

### 3. Compute ROUGE-Lsum and BERTScore

```bash
python compute_rouge_bert_stats.py
```

This script:

- computes per-example ROUGE-Lsum and BERTScore values;
- computes average values for each saved checkpoint;
- adds the average results to `computed_stats/all_stats.json`;
- saves the per-example results separately.

### 4. Compute the statistical results

```bash
python compute_statistical_tests.py
```

This script computes:

- means;
- sample standard deviations;
- 95% confidence intervals;
- paired Wilcoxon signed-rank tests;
- mean and median paired differences;
- unadjusted p-values;
- Holm-adjusted p-values.

The statistical results are added to:

```text
computed_stats/all_stats.json
```

### 5. Generate the figures

```bash
python plot_results.py
```

The generated figures are saved in the `figures/` directory.

## Output Files

### Experiment results

```text
results/
├── summarization_results.json
├── constrained_summary_results.json
└── code_optimization_results.json
```

### Computed statistics

```text
computed_stats/
├── all_stats.json
└── summarization_rouge_bert_by_example.json
```

### Figures

```text
figures/
├── 01_generation_metrics.png
├── 02_average_constraint_completion_ratio.png
├── 03_constrained_category_heatmap.png
└── 04_constrained_pass_rate.png
```

## Model Configuration

All LLM calls use `gemma3:4b` through Ollama with the following settings:

```text
temperature: 0.2
context window: 4096 tokens
maximum generated output: 700 tokens
```

The same model is used for:

- initial generation;
- feedback generation;
- revision;
- constrained summarization evaluation.
