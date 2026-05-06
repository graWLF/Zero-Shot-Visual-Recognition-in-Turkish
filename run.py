"""
Full pipeline runner.

Runs data collection, then inference + evaluation for ViT-B/32 followed by
ViT-L/14, then generates all plots comparing both models.

Usage:
    python run.py              # full pipeline (collect + both models + plots)
    python run.py --skip-collect   # skip data collection (images already downloaded)
    python run.py --model b32      # only run ViT-B/32
    python run.py --model l14      # only run ViT-L/14
"""

import argparse
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
        print(f"\nERROR: step failed with exit code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def collect(args):
    run([PYTHON, "scripts/collect_imagenet.py"],
        "Step 1/3 — Collect animal images (Wikimedia)")
    run([PYTHON, "scripts/collect_food101.py"],
        "Step 2/3 — Collect Food-101 images")
    run([PYTHON, "scripts/collect_wikimedia.py"],
        "Step 3/3 — Collect Wikimedia images (food Turkish, traffic signs, landmarks)")


def run_model(tag, label):
    run([PYTHON, "scripts/inference.py",
         "--model", tag,
         "--output", f"predictions_{tag}.json"],
        f"Inference — CLIP {label}")

    run([PYTHON, "scripts/evaluate.py",
         "--input", f"predictions_{tag}.json",
         "--output", f"evaluation_summary_{tag}.json"],
        f"Evaluate — CLIP {label}")

    # Keep the last model's results as the default files (used by visualize.py)
    import shutil
    results = REPO_ROOT / "results"
    shutil.copy(results / f"predictions_{tag}.json",    results / "predictions.json")
    shutil.copy(results / f"evaluation_summary_{tag}.json", results / "evaluation_summary.json")
    print(f"  Copied {tag} results → predictions.json / evaluation_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection (images already present)")
    parser.add_argument("--model", choices=["b32", "l14", "both"], default="both",
                        help="Which model(s) to run (default: both)")
    args = parser.parse_args()

    if not args.skip_collect:
        collect(args)
    else:
        print("Skipping data collection.")

    models_to_run = {
        "b32": ("b32", "ViT-B/32"),
        "l14": ("l14", "ViT-L/14"),
    }

    if args.model == "both":
        # Run B/32 first so L/14 ends up as the default (current) result
        for tag, label in [models_to_run["b32"], models_to_run["l14"]]:
            run_model(tag, label)
    else:
        tag, label = models_to_run[args.model]
        run_model(tag, label)

    # Visualize expects:
    #   results/predictions.json            → current model (L/14 if both)
    #   results/evaluation_summary.json     → current model
    #   results/predictions_vitb32.json     → B/32 backup (for comparison plot)
    #   results/evaluation_summary_vitb32.json
    #
    # Rename b32 files to the names visualize.py expects
    results = REPO_ROOT / "results"
    for src, dst in [
        ("predictions_b32.json",         "predictions_vitb32.json"),
        ("evaluation_summary_b32.json",  "evaluation_summary_vitb32.json"),
    ]:
        src_path = results / src
        if src_path.exists():
            src_path.rename(results / dst)

    run([PYTHON, "scripts/visualize.py"], "Generate all plots")

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("  Results : results/")
    print("  Plots   : plots/")
    print("="*60)


if __name__ == "__main__":
    main()
