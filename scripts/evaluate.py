import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DOMAINS = ["animals", "food_international", "food_turkish", "traffic_signs", "landmarks"]
CONDITIONS = ["english", "turkish", "mixed"]
BONFERRONI_N = 15
ALPHA = 0.05 / BONFERRONI_N  # 0.0033


def mcnemar_test(b01, b10):
    """McNemar test: b01 = correct in cond A but not B, b10 = correct in B but not A."""
    if b01 + b10 == 0:
        return 1.0
    table = np.array([[0, b01], [b10, 0]])
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return p


def compute_accuracy(records, condition, k=1):
    correct = sum(1 for r in records if r[condition][f"correct_top{k}"])
    return correct / len(records) if records else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="predictions.json",
                        help="Input predictions filename")
    parser.add_argument("--output", type=str, default="evaluation_summary.json",
                        help="Output summary filename")
    args = parser.parse_args()

    pred_path = RESULTS_DIR / args.input
    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    by_domain = defaultdict(list)
    for r in predictions:
        by_domain[r["domain"]].append(r)

    summary = {"per_domain": {}, "mcnemar": []}

    print(f"\n{'Domain':<22} {'Cond':<10} {'Top-1':>6} {'Top-5':>6}  N")
    print("-" * 55)

    for domain in DOMAINS:
        records = by_domain[domain]
        summary["per_domain"][domain] = {}
        for condition in CONDITIONS:
            top1 = compute_accuracy(records, condition, k=1)
            top5 = compute_accuracy(records, condition, k=5)
            summary["per_domain"][domain][condition] = {
                "top1": round(top1, 4),
                "top5": round(top5, 4),
                "n": len(records),
            }
            print(f"{domain:<22} {condition:<10} {top1:>6.3f} {top5:>6.3f}  {len(records)}")
        print()

    print(f"\n{'Domain':<22} {'Comparison':<25} {'p-value':>10} {'Sig*':>6}")
    print("-" * 68)

    comparisons = [
        ("english", "turkish"),
        ("english", "mixed"),
        ("turkish", "mixed"),
    ]

    for domain in DOMAINS:
        records = by_domain[domain]
        for cond_a, cond_b in comparisons:
            b01 = sum(1 for r in records if r[cond_a]["correct_top1"] and not r[cond_b]["correct_top1"])
            b10 = sum(1 for r in records if not r[cond_a]["correct_top1"] and r[cond_b]["correct_top1"])
            p = mcnemar_test(b01, b10)
            sig = "*" if p < ALPHA else ""
            label = f"{cond_a} vs {cond_b}"
            print(f"{domain:<22} {label:<25} {p:>10.4f} {sig:>6}")
            summary["mcnemar"].append({
                "domain": domain,
                "comparison": label,
                "b01": b01,
                "b10": b10,
                "p_value": round(p, 6),
                "significant": bool(p < ALPHA),
                "bonferroni_alpha": ALPHA,
            })
        print()

    out_path = RESULTS_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
