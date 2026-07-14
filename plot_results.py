import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


STATS_FILE = Path("computed_stats/all_stats.json")
OUTPUT_DIR = Path("figures")


ROUND_LABELS = {
    0: "Single-pass",
    1: "Round 1",
    3: "Round 3",
    5: "Round 5",
    10: "Round 10",
    50: "Round 50",
}


def load_stats(path: Path) -> dict:
    """ Loads stats from a JSON file. """
    if not path.exists():
        raise FileNotFoundError(f"Could not find stats file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(filename: str) -> None:
    """ Saves a figure. """
    OUTPUT_DIR.mkdir(exist_ok=True)

    png_path = OUTPUT_DIR / f"{filename}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {png_path}")


def setup_style() -> None:
    """ Sets up the plot design. """
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )


def plot_summarization_and_code_metrics(stats: dict) -> None:
    """ Plots summarization and code metrics. """
    summarization = stats["summarization_similarity_stats"]
    code = stats["code_optimization_similarity_stats"]
    summarization_metrics = stats["summarization_rouge_bert_stats"]

    rounds = [0, 1, 3, 5, 10, 50]
    x = np.arange(len(rounds))
    labels = [ROUND_LABELS[r] for r in rounds]

    summarization_cosine = [
        next(
            row["average_similarity"]
            for row in summarization
            if row["round"] == r
        )
        for r in rounds
    ]

    summarization_rouge = [
        next(
            row["average_rouge_lsum_f1"]
            for row in summarization_metrics
            if row["round"] == r
        )
        for r in rounds
    ]

    summarization_bertscore = [
        next(
            row["average_bertscore_f1"]
            for row in summarization_metrics
            if row["round"] == r
        )
        for r in rounds
    ]

    code_cosine = [
        next(
            row["average_similarity"]
            for row in code
            if row["round"] == r
        )
        for r in rounds
    ]

    plt.figure(figsize=(11, 6.5))

    # the summarization metrics are solid lines
    plt.plot(
        x,
        summarization_cosine,
        marker="o",
        linestyle="-",
        linewidth=2.5,
        label="Summarization: cosine similarity",
    )

    plt.plot(
        x,
        summarization_rouge,
        marker="s",
        linestyle="-",
        linewidth=2.5,
        label="Summarization: ROUGE-Lsum F1",
    )

    plt.plot(
        x,
        summarization_bertscore,
        marker="^",
        linestyle="-",
        linewidth=2.5,
        label="Summarization: BERTScore F1",
    )

    # the code optimization metric is interrupted line
    plt.plot(
        x,
        code_cosine,
        marker="D",
        linestyle="--",
        linewidth=2.5,
        label="Code optimization: cosine similarity",
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Average metric score")

    plt.title(
        "Summarization and Code Optimization Across Refinement Rounds"
    )

    plt.ylim(0.20, 0.93)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_figure("01_generation_metrics")
    plt.close()


def plot_average_constraint_completion_ratio(stats: dict) -> None:
    """ Plots average constraint completion ratio. """
    constrained = stats["constrained_summary_overall_stats"]

    rounds = [0, 1, 3, 5, 10, 50]
    x = np.arange(len(rounds))
    labels = [ROUND_LABELS[r] for r in rounds]

    completion_values = [
        next(
            row["average_constraint_completion_ratio"]
            for row in constrained
            if row["round"] == r
        ) * 100
        for r in rounds
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(x, completion_values, color="gray")

    for bar, value in zip(bars, completion_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value - 3,
            f"{value:.1f}%",
            ha="center",
            va="top",
            fontsize=12,
            color="white",
        )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel("Average completion ratio (%)")
    plt.title("Average Constraint Completion Ratio", pad=15)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_figure("02_average_constraint_completion_ratio")
    plt.close()


def plot_constrained_category_heatmap(stats: dict) -> None:
    """ Plots constrained category heatmap. """
    category_stats = stats["constrained_summary_category_stats"]

    rounds = [0, 1, 3, 5, 10, 50]
    round_labels = [ROUND_LABELS[r] for r in rounds]

    categories = [
        "goal",
        "faithfulness",
        "coverage",
        "tone",
        "style",
        "length",
        "entities",
    ]

    category_labels = [
        "Goal",
        "Faithfulness",
        "Coverage",
        "Tone",
        "Style",
        "Length",
        "Entities",
    ]

    heatmap_values = []

    for category in categories:
        row_values = []

        for r in rounds:
            row = next(item for item in category_stats if item["round"] == r)
            pass_rate = row.get(f"{category}_pass_rate")

            if pass_rate is None:
                row_values.append(np.nan)
            else:
                row_values.append(pass_rate * 100)

        heatmap_values.append(row_values)

    heatmap_values = np.array(heatmap_values)

    yellow_to_midtone_green = LinearSegmentedColormap.from_list(
        "yellow_to_midtone_green",
        [
            "#fff176",
            "#eceb78",
            "#cfdc7a",
            "#9fc77d",
            "#6fa58e",
        ],
    )

    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        heatmap_values,
        aspect="auto",
        vmin=50,
        vmax=100,
        cmap=yellow_to_midtone_green,
    )

    plt.xticks(np.arange(len(rounds)), round_labels, rotation=25, ha="right")
    plt.yticks(np.arange(len(categories)), category_labels)

    for i in range(len(categories)):
        for j in range(len(rounds)):
            value = heatmap_values[i, j]

            if not np.isnan(value):
                plt.text(
                    j,
                    i,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                )

    cbar = plt.colorbar(image)
    cbar.set_label("Pass rate (%)")

    plt.title("Category-Specific Pass Rates")
    plt.tight_layout()

    save_figure("03_constrained_category_heatmap")
    plt.close()


def plot_constrained_pass_rate(stats: dict) -> None:
    """ Plots constrained pass rate. """
    constrained = stats["constrained_summary_overall_stats"]

    rounds = [0, 1, 3, 5, 10, 50]
    x = np.arange(len(rounds))
    labels = [ROUND_LABELS[r] for r in rounds]

    pass_rates = [
        next(row["overall_pass_rate"] for row in constrained if row[
            "round"] == r) * 100
        for r in rounds
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(x, pass_rates)

    for bar, value in zip(bars, pass_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Overall pass rate (%)")
    plt.title("Overall Pass Rate Across Refinement Rounds")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_figure("04_constrained_pass_rate")
    plt.close()


def main() -> None:
    """ Main function. """
    setup_style()

    stats = load_stats(STATS_FILE)

    plot_summarization_and_code_metrics(stats)
    plot_average_constraint_completion_ratio(stats)
    plot_constrained_category_heatmap(stats)
    plot_constrained_pass_rate(stats)

    print("\nPNG figures saved in the 'figures' folder.")


if __name__ == "__main__":
    main()
