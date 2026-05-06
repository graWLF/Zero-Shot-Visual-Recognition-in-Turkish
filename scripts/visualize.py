import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
DOMAINS = ["animals", "food_international", "food_turkish", "traffic_signs", "landmarks"]
CONDITIONS = ["english", "turkish", "mixed"]
COLORS = {"english": "#4C72B0", "turkish": "#DD8452", "mixed": "#55A868"}

FRAG_BY_DOMAIN = {
    "traffic_signs":      1.419,
    "landmarks":          1.105,
    "food_international": 1.084,
    "animals":            1.027,
    "food_turkish":       0.693,
}


def load_data():
    with open(RESULTS_DIR / "predictions.json", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(RESULTS_DIR / "evaluation_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    return predictions, summary


def load_b32_data():
    path = RESULTS_DIR / "predictions_vitb32.json"
    if not path.exists():
        return None, None
    with open(path, encoding="utf-8") as f:
        predictions = json.load(f)
    path2 = RESULTS_DIR / "evaluation_summary_vitb32.json"
    with open(path2, encoding="utf-8") as f:
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
    by_class = defaultdict(lambda: {"en_correct": 0, "tr_correct": 0, "total": 0})
    for r in predictions:
        key = (r["domain"], r["true_label_en"])
        by_class[key]["en_correct"] += int(r["english"]["correct_top1"])
        by_class[key]["tr_correct"] += int(r["turkish"]["correct_top1"])
        by_class[key]["total"] += 1

    domain_colors = {d: plt.cm.tab10(i) for i, d in enumerate(DOMAINS)}
    xs, ys, cs = [], [], []
    for (domain, cls_en), data in by_class.items():
        n = data["total"]
        if n == 0:
            continue
        drop = data["en_correct"] / n - data["tr_correct"] / n
        xs.append(FRAG_BY_DOMAIN[domain])
        ys.append(drop)
        cs.append(domain_colors[domain])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, c=cs, alpha=0.75, s=60)

    if len(xs) > 1:
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        m, b = np.polyfit(xs_arr, ys_arr, 1)
        xline = np.linspace(min(xs_arr), max(xs_arr), 100)
        ax.plot(xline, m * xline + b, "k--", linewidth=1.2, label=f"y={m:.2f}x+{b:.2f}")

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
    """
    Replaces the unreadable 60×60 matrix with two focused panels:
    - Left:  horizontal bar chart of top-20 misclassification pairs
    - Right: compact heatmap of only the classes involved in those pairs
    """
    from scripts.prompts import CLASSES

    en_to_tr = {}
    cls_to_domain = {}
    domain_colors_map = {d: plt.cm.tab10(i) for i, d in enumerate(DOMAINS)}
    for domain in DOMAINS:
        for cls in CLASSES[domain]:
            en_to_tr[cls["en"]] = cls["tr"]
            cls_to_domain[cls["en"]] = domain

    # Count off-diagonal misclassifications (Turkish condition)
    pairs = defaultdict(int)
    for r in predictions:
        true_en = r["true_label_en"]
        pred_en = r["turkish"]["top1"]
        if true_en != pred_en:
            pairs[(true_en, pred_en)] += 1

    top_n = 20
    top_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:top_n]

    # ── Panel 1: horizontal bar chart ────────────────────────────────────────
    labels, counts, bar_colors = [], [], []
    for (true_en, pred_en), count in top_pairs:
        true_tr  = en_to_tr.get(true_en, true_en)
        pred_tr  = en_to_tr.get(pred_en, pred_en)
        domain   = cls_to_domain.get(true_en, "?")
        labels.append(f"{true_tr}  →  {pred_tr}")
        counts.append(count)
        bar_colors.append(domain_colors_map[domain])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    ax1 = axes[0]
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, counts, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of images misclassified this way", fontsize=10)
    ax1.set_title("Top 20 Misclassification Pairs\n(Turkish prompts — true → predicted)", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=9)

    # Domain legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=domain_colors_map[d], alpha=0.85, label=d.replace("_", " "))
               for d in DOMAINS]
    ax1.legend(handles=handles, fontsize=8, title="Domain", loc="lower right")

    # ── Panel 2: compact heatmap of only involved classes ────────────────────
    involved_true = [p[0] for p, _ in top_pairs]
    involved_pred = [p[1] for p, _ in top_pairs]
    involved = sorted(set(involved_true + involved_pred),
                      key=lambda x: en_to_tr.get(x, x))

    n = len(involved)
    cls_idx = {c: i for i, c in enumerate(involved)}
    matrix = np.zeros((n, n), dtype=int)
    for (true_en, pred_en), count in pairs.items():
        if true_en in cls_idx and pred_en in cls_idx:
            matrix[cls_idx[true_en], cls_idx[pred_en]] += count

    tr_labels = [en_to_tr.get(c, c) for c in involved]
    ax2 = axes[1]
    sns.heatmap(matrix, xticklabels=tr_labels, yticklabels=tr_labels,
                cmap="YlOrRd", ax=ax2, annot=True, fmt="d",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="#DDDDDD",
                cbar_kws={"shrink": 0.7})
    ax2.set_xlabel("Predicted (Turkish)", fontsize=10)
    ax2.set_ylabel("True (Turkish)", fontsize=10)
    ax2.set_title("Confusion — Classes Involved\nin Top-20 Mistakes", fontsize=11, fontweight="bold")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

    plt.suptitle("Where Does CLIP Go Wrong with Turkish Prompts?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrix.png")


def plot_top_misclassifications(predictions, top_n=20):
    from scripts.prompts import CLASSES

    en_to_tr = {}
    cls_to_domain = {}
    for domain in DOMAINS:
        for cls in CLASSES[domain]:
            en_to_tr[cls["en"]] = cls["tr"]
            cls_to_domain[cls["en"]] = domain

    pairs = defaultdict(int)
    for r in predictions:
        true_en = r["true_label_en"]
        pred_en = r["turkish"]["top1"]
        if true_en != pred_en:
            pairs[(true_en, pred_en)] += 1

    top_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:top_n]

    rows = []
    for (true_en, pred_en), count in top_pairs:
        true_tr = en_to_tr.get(true_en, true_en)
        pred_tr = en_to_tr.get(pred_en, pred_en)
        domain  = cls_to_domain.get(true_en, "?")
        rows.append([domain.replace("_", " "), true_en, true_tr, pred_en, pred_tr, count])

    col_labels = ["Domain", "True (EN)", "True (TR)", "Predicted (EN)", "Predicted (TR)", "Count"]

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(rows) + 1.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F2F2")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")

    ax.set_title(f"Top {top_n} Misclassifications — Turkish Condition (ViT-L/14)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_misclassifications.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved top_misclassifications.png")


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
    from scripts.prompts import CLASSES

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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


def plot_model_comparison(b32_summary, l14_summary):
    """Bar chart comparing EN→TR accuracy gap for ViT-B/32 vs ViT-L/14 per domain."""
    b32_gaps = [
        b32_summary["per_domain"][d]["english"]["top1"] - b32_summary["per_domain"][d]["turkish"]["top1"]
        for d in DOMAINS
    ]
    l14_gaps = [
        l14_summary["per_domain"][d]["english"]["top1"] - l14_summary["per_domain"][d]["turkish"]["top1"]
        for d in DOMAINS
    ]
    b32_en  = [b32_summary["per_domain"][d]["english"]["top1"] for d in DOMAINS]
    l14_en  = [l14_summary["per_domain"][d]["english"]["top1"] for d in DOMAINS]
    b32_tr  = [b32_summary["per_domain"][d]["turkish"]["top1"] for d in DOMAINS]
    l14_tr  = [l14_summary["per_domain"][d]["turkish"]["top1"] for d in DOMAINS]

    x = np.arange(len(DOMAINS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: EN→TR gap per model
    ax = axes[0]
    bars_b = ax.bar(x - width / 2, b32_gaps, width, label="ViT-B/32", color="#4C72B0", alpha=0.85)
    bars_l = ax.bar(x + width / 2, l14_gaps, width, label="ViT-L/14", color="#C44E52", alpha=0.85)
    for bar, v in zip(bars_b, b32_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars_l, l14_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in DOMAINS], fontsize=9)
    ax.set_ylabel("Accuracy Gap (English − Turkish)")
    ax.set_ylim(0, 0.85)
    ax.set_title("EN→TR Accuracy Gap: Does Scaling Help?\n(larger bar = larger gap = scaling hurts Turkish more)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: EN and TR accuracy side by side for both models
    ax2 = axes[1]
    w = 0.2
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    series = [
        (b32_en, "B/32 EN", "#4C72B0", 0.95),
        (b32_tr, "B/32 TR", "#DD8452", 0.95),
        (l14_en, "L/14 EN", "#4C72B0", 0.5),
        (l14_tr, "L/14 TR", "#DD8452", 0.5),
    ]
    for (vals, label, color, alpha), offset in zip(series, offsets):
        bars = ax2.bar(x + offset, vals, w, label=label, color=color, alpha=alpha)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace("_", "\n") for d in DOMAINS], fontsize=9)
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_ylim(0, 1.10)
    ax2.set_title("EN vs TR Accuracy per Model\n(solid = B/32, faded = L/14)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Scaling Up CLIP Does Not Close the Turkish Language Gap",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_comparison.png")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    predictions, summary = load_data()
    b32_predictions, b32_summary = load_b32_data()

    plot_accuracy_barplot(summary)
    plot_fragmentation_scatter(predictions, summary)
    plot_confusion_matrix(predictions)
    plot_top_misclassifications(predictions, top_n=20)
    plot_accuracy_heatmap(summary)
    plot_token_count_barplot()

    if b32_summary is not None:
        plot_model_comparison(b32_summary, summary)
    else:
        print("Skipped model_comparison.png (predictions_vitb32.json not found)")

    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
