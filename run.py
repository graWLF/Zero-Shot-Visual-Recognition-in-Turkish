"""
Full pipeline runner.

Collects images (skips classes already downloaded), runs inference and
evaluation for ViT-B/32 then ViT-L/14, and generates all comparison plots.

Usage:
    python run.py                  # full pipeline (recommended)
    python run.py --model b32      # only ViT-B/32
    python run.py --model l14      # only ViT-L/14
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
PYTHON = sys.executable


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: step failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def collect():
    run([PYTHON, "scripts/collect_imagenet.py"],
        "Collect animal images (Wikimedia — skips existing)")
    run([PYTHON, "scripts/collect_food101.py"],
        "Collect Food-101 images (skips existing)")
    run([PYTHON, "scripts/collect_wikimedia.py"],
        "Collect Wikimedia images — food Turkish, traffic signs, landmarks (skips existing)")


def run_model(tag, label):
    run([PYTHON, "scripts/inference.py",
         "--model", tag,
         "--output", f"predictions_{tag}.json"],
        f"Inference — CLIP {label}")

    run([PYTHON, "scripts/evaluate.py",
         "--input",  f"predictions_{tag}.json",
         "--output", f"evaluation_summary_{tag}.json"],
        f"Evaluate — CLIP {label}")

    # Keep the latest model's files as the default (used by visualize.py)
    results = REPO_ROOT / "results"
    shutil.copy(results / f"predictions_{tag}.json",
                results / "predictions.json")
    shutil.copy(results / f"evaluation_summary_{tag}.json",
                results / "evaluation_summary.json")
    print(f"  Copied {tag} results → predictions.json / evaluation_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["b32", "l14", "both"], default="both",
                        help="Which model(s) to run (default: both)")
    args = parser.parse_args()

    # Always run collect — each script skips classes that already have 20 images
    collect()

    if args.model == "both":
        # B/32 first so L/14 ends up as the default result
        run_model("b32", "ViT-B/32")
        run_model("l14", "ViT-L/14")
    else:
        labels = {"b32": "ViT-B/32", "l14": "ViT-L/14"}
        run_model(args.model, labels[args.model])

    # Rename b32 files to the names visualize.py expects for the comparison plot
    results = REPO_ROOT / "results"
    for src, dst in [
        ("predictions_b32.json",        "predictions_vitb32.json"),
        ("evaluation_summary_b32.json", "evaluation_summary_vitb32.json"),
    ]:
        src_path = results / src
        if src_path.exists():
            shutil.copy(src_path, results / dst)

    run([PYTHON, "scripts/visualize.py"], "Generate all plots")

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("  Results : results/")
    print("  Plots   : plots/")
    print("="*60)


if __name__ == "__main__":
    main()
