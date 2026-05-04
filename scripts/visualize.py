import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

matplotlib.rcParams["font.family"] = "DejaVu Sans"

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
DOMAINS = ["animals", "food_international", "food_turkish", "traffic_signs", "landmarks"]
CONDITIONS = ["english", "turkish", "mixed"]
COLORS = {"english": "#4C72B0", "turkish": "#DD8452", "mixed": "#55A868"}

FRAG_BY_DOMAIN = {
    "traffic_signs":     1.419,
    "landmarks":         1.105,
    "food_international":1.084,
    "animals":           1.027,
    "food_turkish":      0.693,
}


def load_data():
    with open(RESULTS_DIR / "predictions.json", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(RESULTS_DIR / "evaluation_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    return predictions, summary


def plot_accuracy_barplot(summary):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DOMAINS))
    width = 0.25

    for i, condition in enumerate(CONDITIONS):
        vals = [summary["per_domain"][d][condition]["top1"] for d in DOMAINS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=condition.capitalize(),
                      color=COLORS[condition], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in DOMAINS], fontsize=9)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("CLIP Zero-Shot Top-1 Accuracy by Domain and Prompt Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_barplot.png", dpi=150)
    plt.close()
    print("Saved accuracy_barplot.png")


def plot_fragmentation_scatter(predictions, summary):
    by_class = defaultdict(lambda: {"en_correct": 0, "tr_correct": 0, "total": 0, "domain": ""})
    for r in predictions:
        key = (r["domain"], r["true_label_en"])
        by_class[key]["en_correct"] += int(r["english"]["correct_top1"])
        by_class[key]["tr_correct"] += int(r["turkish"]["correct_top1"])
        by_class[key]["total"] += 1
        by_class[key]["domain"] = r["domain"]

    domain_colors = {d: plt.cm.tab10(i) for i, d in enumerate(DOMAINS)}
    xs, ys, cs, labels = [], [], [], []
    for (domain, cls_en), data in by_class.items():
        n = data["total"]
        if n == 0:
            continue
        acc_en = data["en_correct"] / n
        acc_tr = data["tr_correct"] / n
        drop = acc_en - acc_tr
        frag = FRAG_BY_DOMAIN[domain]
        xs.append(frag)
        ys.append(drop)
        cs.append(domain_colors[domain])
        labels.append(cls_en)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, c=cs, alpha=0.75, s=60)

    if len(xs) > 1:
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        m, b = np.polyfit(xs_arr, ys_arr, 1)
        xline = np.linspace(min(xs_arr), max(xs_arr), 100)
        ax.plot(xline, m * xline + b, "k--", linewidth=1.2, label=f"y={m:.2f}x+{b:.2f}")
        ax.legend()

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=domain_colors[d], markersize=8, label=d)
               for d in DOMAINS]
    ax.legend(handles=handles, fontsize=8, title="Domain")
    ax.set_xlabel("Avg BPE Fragmentation Rate (domain)")
    ax.set_ylabel("Accuracy Drop (English − Turkish)")
    ax.set_title("BPE Fragmentation vs Accuracy Drop per Class")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fragmentation_scatter.png", dpi=150)
    plt.close()
    print("Saved fragmentation_scatter.png")


def plot_confusion_matrix(predictions):
    turkish_preds = [(r["true_label_tr"], r["turkish"]["top1"]) for r in predictions]

    from scripts.prompts import CLASSES
    tr_labels = []
    en_to_tr = {}
    for domain in DOMAINS:
        for cls in CLASSES[domain]:
            if cls["tr"] not in tr_labels:
                tr_labels.append(cls["tr"])
            en_to_tr[cls["en"]] = cls["tr"]

    label_set = sorted(set(lbl for lbl, _ in turkish_preds))
    label_to_i = {l: i for i, l in enumerate(label_set)}
    n = len(label_set)
    matrix = np.zeros((n, n), dtype=int)

    for true_tr, pred_en in turkish_preds:
        pred_tr = en_to_tr.get(pred_en, pred_en)
        i = label_to_i.get(true_tr, -1)
        j = label_to_i.get(pred_tr, -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(12, n * 0.4), max(10, n * 0.35)))
    sns.heatmap(matrix, xticklabels=label_set, yticklabels=label_set,
                cmap="Blues", ax=ax, linewidths=0.3, linecolor="gray",
                annot=(n <= 30), fmt="d", annot_kws={"size": 5})
    ax.set_xlabel("Predicted (Turkish)")
    ax.set_ylabel("True (Turkish)")
    ax.set_title("Confusion Matrix — Turkish Condition")
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


def plot_accuracy_heatmap(summary):
    data = np.array([[summary["per_domain"][d][c]["top1"] for c in CONDITIONS]
                     for d in DOMAINS])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(data, xticklabels=[c.capitalize() for c in CONDITIONS],
                yticklabels=[d.replace("_", " ").title() for d in DOMAINS],
                cmap="YlGn", ax=ax, annot=True, fmt=".2f", linewidths=0.5,
                vmin=0, vmax=1)
    ax.set_title("Top-1 Accuracy Heatmap (Domain × Condition)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_heatmap.png", dpi=150)
    plt.close()
    print("Saved accuracy_heatmap.png")


def plot_token_count_barplot():
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    from scripts.prompts import CLASSES
    class_names, en_counts, tr_counts, domain_labels = [], [], [], []
    for domain in DOMAINS:
        for cls in CLASSES[domain]:
            class_names.append(cls["en"])
            en_counts.append(len(tokenizer.encode(cls["en"], add_special_tokens=False)))
            tr_counts.append(len(tokenizer.encode(cls["tr"], add_special_tokens=False)))
            domain_labels.append(domain)

    n = len(class_names)
    x = np.arange(n)
    width = 0.4
    domain_colors = {d: plt.cm.tab10(i) for i, d in enumerate(DOMAINS)}

    fig, ax = plt.subplots(figsize=(max(16, n * 0.35), 6))
    for i, (en, tr, domain) in enumerate(zip(en_counts, tr_counts, domain_labels)):
        color = domain_colors[domain]
        ax.bar(x[i] - width / 2, en, width, color=color, alpha=0.9)
        ax.bar(x[i] + width / 2, tr, width, color=color, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Token Count")
    ax.set_title("BPE Token Count per Class: English (solid) vs Turkish (faded)")
    ax.set_ylim(0, max(max(en_counts), max(tr_counts)) + 2)
    ax.grid(axis="y", alpha=0.3)

    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=domain_colors[d], markersize=10, label=d)
               for d in DOMAINS]
    handles += [
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.9, label="English"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4, label="Turkish"),
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "token_count_barplot.png", dpi=150)
    plt.close()
    print("Saved token_count_barplot.png")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    predictions, summary = load_data()
    plot_accuracy_barplot(summary)
    plot_fragmentation_scatter(predictions, summary)
    plot_confusion_matrix(predictions)
    plot_accuracy_heatmap(summary)
    plot_token_count_barplot()
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
