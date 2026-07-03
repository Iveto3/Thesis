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
    if not path.exists():
        raise FileNotFoundError(f"Could not find stats file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    png_path = OUTPUT_DIR / f"{filename}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {png_path}")


def setup_style() -> None:
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


def plot_similarity_trends(stats: dict) -> None:
    summarization = stats["summarization_similarity_stats"]
    code = stats["code_optimization_similarity_stats"]

    rounds = [0, 1, 3, 5, 10, 50]
    x = np.arange(len(rounds))
    labels = [ROUND_LABELS[r] for r in rounds]

    summarization_values = [
        next(row["average_similarity"] for row in summarization if row[
            "round"] == r)
        for r in rounds
    ]

    code_values = [
        next(row["average_similarity"] for row in code if row["round"] == r)
        for r in rounds
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        summarization_values,
        marker="o",
        linewidth=2.5,
        label="Summarization",
    )

    plt.plot(
        x,
        code_values,
        marker="o",
        linewidth=2.5,
        label="Code optimization",
    )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Average cosine similarity")
    plt.title("Cosine Similarity Across Refinement Rounds")
    plt.ylim(0.65, 0.93)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_figure("01_similarity_trends")
    plt.close()


def plot_constrained_pass_rate(stats: dict) -> None:
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
    plt.title("Constraint Pass Rate Across Refinement Rounds")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_figure("02_constrained_pass_rate")
    plt.close()


def plot_constrained_category_heatmap(stats: dict) -> None:
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

    # Yellow for lower observed values, midtone green for 100%.
    # vmin=50 makes the 58% style score appear yellow instead of already green.
    yellow_to_midtone_green = LinearSegmentedColormap.from_list(
        "yellow_to_midtone_green",
        [
            "#fff176",  # yellow
            "#eceb78",  # yellow-green
            "#cfdc7a",  # soft yellow-green
            "#9fc77d",  # light/midtone green
            "#6fa58e",  # midtone green for 100%
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

    plt.title("Category-Level Constraint Pass Rates")
    plt.tight_layout()

    save_figure("03_constrained_category_heatmap")
    plt.close()


def main() -> None:
    setup_style()

    stats = load_stats(STATS_FILE)

    plot_similarity_trends(stats)
    plot_constrained_pass_rate(stats)
    plot_constrained_category_heatmap(stats)

    print("\nDone. PNG figures saved in the 'figures' folder.")


if __name__ == "__main__":
    main()
