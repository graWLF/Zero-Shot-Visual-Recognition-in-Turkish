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


VIT_TAG = {"b32": "vitb32", "l14": "vitl14"}


def run_model(tag, label):
    vit = VIT_TAG[tag]
    run([PYTHON, "scripts/inference.py",
         "--model", tag,
         "--output", f"predictions_{vit}.json"],
        f"Inference — CLIP {label}")

    run([PYTHON, "scripts/evaluate.py",
         "--input",  f"predictions_{vit}.json",
         "--output", f"evaluation_summary_{vit}.json"],
        f"Evaluate — CLIP {label}")

    # Keep the latest model's files as the default (used by visualize.py)
    results = REPO_ROOT / "results"
    shutil.copy(results / f"predictions_{vit}.json",
                results / "predictions.json")
    shutil.copy(results / f"evaluation_summary_{vit}.json",
                results / "evaluation_summary.json")
    print(f"  Copied {vit} results → predictions.json / evaluation_summary.json")


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

    run([PYTHON, "scripts/visualize.py"], "Generate all plots")

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("  Results : results/")
    print("  Plots   : plots/")
    print("="*60)


if __name__ == "__main__":
    main()
