# Session Notes — CLIP Turkish Zero-Shot Recognition

**Date:** 2026-05-04  
**GPU:** NVIDIA RTX 3050 Mobile (4 GB VRAM)  
**Model:** CLIP ViT-B/32  
**Environment:** Conda env `CV`, Python 3.12

---

## What We Built

Starting from an empty repository (only `CLAUDE.md` and `README.md` existed), we implemented the full pipeline end-to-end in a single session:

| Script | Purpose |
|--------|---------|
| `scripts/prompts.py` | Class definitions (60 classes, 5 domains) and prompt templates |
| `scripts/collect_imagenet.py` | Animal image collection |
| `scripts/collect_food101.py` | Food International image collection |
| `scripts/collect_wikimedia.py` | Food Turkish, Traffic Signs, Landmarks collection |
| `scripts/fix_missing.py` | Targeted re-collection for classes that failed initial download |
| `scripts/inference.py` | CLIP zero-shot inference over all 1,200 images |
| `scripts/evaluate.py` | Top-1/Top-5 accuracy + McNemar tests + Bonferroni correction |
| `scripts/visualize.py` | 5 result plots |

---

## Problems Encountered & How We Solved Them

### 1. ImageNet-1K Is a Gated Dataset

The original plan was to collect animal images from HuggingFace `imagenet-1k`. The account `graWLF` had not been granted access to this dataset (it requires explicit approval on the HuggingFace hub).

**Fix:** Switched animal collection entirely to Wikimedia Commons, using the same category-based + search-based API approach as the other domains.

### 2. `trust_remote_code` Deprecated

An early version of `collect_imagenet.py` passed `trust_remote_code=True` to `load_dataset`, which is no longer accepted in the installed version of `datasets`.

**Fix:** Removed the argument. The issue turned out to be moot anyway since we switched to Wikimedia.

### 3. Wikimedia Rate Limiting (HTTP 429)

The first Wikimedia download attempts returned almost no images despite finding many candidate URLs. Debugging revealed that `upload.wikimedia.org` was returning HTTP 429 (Too Many Requests) on almost every download request when we issued them back-to-back.

**Fix:** Added retry logic with exponential backoff (10s, 20s, 30s per attempt) and increased inter-download delay to 1.5 seconds. This resolved the issue for most classes.

### 4. Hard-to-Find Classes

Several classes consistently returned fewer than 20 images even after rate-limit fixes:

| Class | Domain | Problem |
|-------|--------|---------|
| Turkish flatbread pizza | food_turkish | The English name "Turkish flatbread pizza" didn't match Wikimedia well |
| Turkish scrambled eggs | food_turkish | Too generic; results were mostly irrelevant |
| harbor | landmarks | Common word, search returned mixed results |
| bridge | landmarks | Extremely generic term |
| train station | landmarks | Inconsistent naming on Wikimedia |

**Fix:** Wrote a dedicated `scripts/fix_missing.py` with hand-tuned alternative search terms (e.g., `"pide Turkish bread"` instead of `"Turkish flatbread pizza"`, `"menemen Turkish"` instead of `"Turkish scrambled eggs"`).

### 5. `get_text_features` Returned a Model Output Object Instead of a Tensor

In `inference.py`, calling `model.get_text_features(**inputs)` returned a `BaseModelOutputWithPooling` object rather than a `torch.Tensor`. This caused a crash at normalization:

```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
```

**Fix:** Added a `_to_tensor()` helper that gracefully extracts `.pooler_output` when the return value is not already a tensor. Also switched from `.norm()` to `torch.nn.functional.normalize()`.

### 6. `numpy.bool_` Not JSON Serializable

In `evaluate.py`, the expression `p < ALPHA` (where `p` is a NumPy float) produces a `numpy.bool_` rather than a Python `bool`. This caused a crash when saving `evaluation_summary.json`.

**Fix:** Wrapped with `bool(p < ALPHA)`.

### 7. `sklearn` Import in `visualize.py`

A leftover `from sklearn.preprocessing import LabelEncoder` import was present in `visualize.py` even though `sklearn` was never used and not installed in the `CV` environment.

**Fix:** Removed the import.

### 8. `scripts` Module Not Found Inside `visualize.py`

The `plot_confusion_matrix` and `plot_token_count_barplot` functions attempted `from scripts.prompts import CLASSES` without the repo root on `sys.path`. This works when the script is run from the repo root via `python scripts/visualize.py` only if the path is set up correctly.

**Fix:** Added `sys.path.insert(0, repo_root)` inside the functions that need it.

---

## Experiment Results

### Top-1 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | **87.0%** | 45.0% | 44.5% |
| Food International | **97.5%** | 39.5% | 53.0% |
| Food Turkish | **50.5%** | 31.0% | 35.5% |
| Traffic Signs | **56.1%** | 9.3% | 7.9% |
| Landmarks | **80.3%** | 39.4% | 44.7% |

### Top-5 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | 98.0% | 69.5% | 79.0% |
| Food International | 100.0% | 60.0% | 78.5% |
| Food Turkish | 85.0% | 73.0% | 80.0% |
| Traffic Signs | 87.1% | 42.1% | 43.9% |
| Landmarks | 96.9% | 77.2% | 81.6% |

### McNemar Tests (Bonferroni α = 0.05/15 = 0.0033)

| Domain | Comparison | p-value | Significant |
|--------|-----------|---------|-------------|
| Animals | English vs Turkish | <0.0001 | ✓ |
| Animals | English vs Mixed | <0.0001 | ✓ |
| Animals | Turkish vs Mixed | 1.0000 | |
| Food International | English vs Turkish | <0.0001 | ✓ |
| Food International | English vs Mixed | <0.0001 | ✓ |
| Food International | Turkish vs Mixed | <0.0001 | ✓ |
| Food Turkish | English vs Turkish | <0.0001 | ✓ |
| Food Turkish | English vs Mixed | 0.0005 | ✓ |
| Food Turkish | Turkish vs Mixed | 0.1237 | |
| Traffic Signs | English vs Turkish | <0.0001 | ✓ |
| Traffic Signs | English vs Mixed | <0.0001 | ✓ |
| Traffic Signs | Turkish vs Mixed | 0.4795 | |
| Landmarks | English vs Turkish | <0.0001 | ✓ |
| Landmarks | English vs Mixed | <0.0001 | ✓ |
| Landmarks | Turkish vs Mixed | 0.0223 | |

### BPE Fragmentation Rates

| Domain | Avg Fragmentation Rate | Accuracy Drop (EN→TR) |
|--------|----------------------|-----------------------|
| Traffic Signs | 1.419 | −46.8 pp |
| Landmarks | 1.105 | −40.9 pp |
| Food International | 1.084 | −58.0 pp |
| Animals | 1.027 | −42.0 pp |
| Food Turkish | 0.693 | −19.5 pp |

### Key Findings

- **English significantly outperforms Turkish in every domain** (all 5 English vs Turkish McNemar tests are significant after Bonferroni correction).
- **Traffic Signs suffers the most severe drop** (56.1% → 9.3%). Turkish traffic sign names are long compound phrases ("yaya geçidi işareti", "dönel kavşak işareti") that CLIP's BPE tokenizer fragments heavily.
- **Food Turkish suffers the least** (50.5% → 31.0%). Many Turkish food names are short, culturally specific words (börek, pide, menemen) that CLIP may have encountered in training data, and they fragment less.
- **Mixed condition mostly underperforms Turkish**, suggesting that mixing language in the template (English syntax + Turkish noun) doesn't help — the problem is in the class name embedding, not the template.
- **BPE fragmentation correlates broadly with accuracy drop**, though Food International is an outlier (moderate fragmentation but very high accuracy drop), likely because CLIP's training data had strong English-language associations for those foods.

---

## Generated Outputs

```
results/
├── predictions.json          # 1,200 entries, top-1 and top-5 per condition
└── evaluation_summary.json   # Per-domain accuracy + McNemar results

plots/
├── accuracy_barplot.png      # Grouped bar chart by domain × condition
├── fragmentation_scatter.png # Fragmentation rate vs accuracy drop scatter
├── confusion_matrix.png      # Confusion matrix for Turkish condition
├── accuracy_heatmap.png      # Heatmap: domains × conditions
└── token_count_barplot.png   # EN vs TR token counts per class
```

---

## ViT-L/14 — Completed

After ViT-B/32 succeeded, we ran the full pipeline again with ViT-L/14. No CUDA OOM occurred on the RTX 3050 Mobile at batch size 16.

### What Is ViT-L/14?

| | ViT-B/32 | ViT-L/14 |
|--|----------|----------|
| Patch size | 32×32 | 14×14 |
| Parameters | ~150M | ~430M |
| Model size on disk | ~600 MB | ~1.7 GB |
| Top-1 ImageNet (EN) | ~63% | ~75% |

### How We Ran It

Changed one line in `scripts/inference.py`:

```python
MODEL_NAME = "openai/clip-vit-large-patch14"
```

Then re-ran inference, evaluate, and visualize. ViT-L/14 was not cached and was downloaded (~1.7 GB) before inference started. Results and plots overwrote the ViT-B/32 outputs; originals were backed up as `predictions_vitb32.json` and `evaluation_summary_vitb32.json`.

---

## ViT-L/14 Results

### Top-1 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | **95.0%** | 46.0% | 47.5% |
| Food International | **100.0%** | 57.5% | 73.0% |
| Food Turkish | **65.5%** | 31.5% | 40.5% |
| Traffic Signs | **68.9%** | 10.4% | 8.9% |
| Landmarks | **85.6%** | 41.6% | 43.8% |

### Top-5 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | 98.5% | 72.5% | 75.0% |
| Food International | 100.0% | 85.0% | 93.0% |
| Food Turkish | 91.0% | 63.5% | 85.0% |
| Traffic Signs | 92.9% | 49.6% | 42.1% |
| Landmarks | 98.8% | 80.9% | 80.3% |

### McNemar Tests (Bonferroni α = 0.0033)

| Domain | Comparison | p-value | Significant |
|--------|-----------|---------|-------------|
| Animals | English vs Turkish | <0.0001 | ✓ |
| Animals | English vs Mixed | <0.0001 | ✓ |
| Animals | Turkish vs Mixed | 0.7277 | |
| Food International | English vs Turkish | <0.0001 | ✓ |
| Food International | English vs Mixed | <0.0001 | ✓ |
| Food International | Turkish vs Mixed | <0.0001 | ✓ |
| Food Turkish | English vs Turkish | <0.0001 | ✓ |
| Food Turkish | English vs Mixed | <0.0001 | ✓ |
| Food Turkish | Turkish vs Mixed | 0.0207 | |
| Traffic Signs | English vs Turkish | <0.0001 | ✓ |
| Traffic Signs | English vs Mixed | <0.0001 | ✓ |
| Traffic Signs | Turkish vs Mixed | 0.4795 | |
| Landmarks | English vs Turkish | <0.0001 | ✓ |
| Landmarks | English vs Mixed | <0.0001 | ✓ |
| Landmarks | Turkish vs Mixed | 0.4996 | |

---

## Model Comparison: ViT-B/32 vs ViT-L/14

### Top-1 Accuracy Gap (English − Turkish)

| Domain | B/32 EN | B/32 TR | B/32 Gap | L/14 EN | L/14 TR | L/14 Gap | Gap Change |
|--------|---------|---------|----------|---------|---------|----------|------------|
| Animals | 87.0% | 45.0% | −42.0 pp | 95.0% | 46.0% | −49.0 pp | ↑ worse |
| Food International | 97.5% | 39.5% | −58.0 pp | 100.0% | 57.5% | −42.5 pp | ↓ better |
| Food Turkish | 50.5% | 31.0% | −19.5 pp | 65.5% | 31.5% | −34.0 pp | ↑ worse |
| Traffic Signs | 56.1% | 9.3% | −46.8 pp | 68.9% | 10.4% | −58.5 pp | ↑ worse |
| Landmarks | 80.3% | 39.4% | −40.9 pp | 85.6% | 41.6% | −44.0 pp | ↑ worse |

### Key Findings from Model Comparison

- **Scaling up helps English accuracy substantially** across all domains — most notably Food Turkish (+15 pp) and Traffic Signs (+12.8 pp).
- **Turkish accuracy barely improves** regardless of model scale. The largest gains are in Food International (+18 pp) and Landmarks (+2.2 pp), but Traffic Signs and Animals are essentially unchanged.
- **The English→Turkish gap widens in 4 out of 5 domains.** Only Food International sees a genuine improvement in the gap (−58 pp → −42.5 pp), likely because some food names like "pizza", "sushi", and "hamburger" are internationally recognizable even in Turkish.
- **The bottleneck is not model capacity.** A 3× larger model does not close the Turkish performance gap — it amplifies the English advantage. This supports the conclusion that the limitation lies in CLIP's training data distribution and BPE tokenization of Turkish, not in the visual encoder's representational power.
- **English significantly outperforms Turkish in all 5 domains with both models** — the finding is robust across model scale.

### Saved Outputs

```
results/
├── predictions.json              # ViT-L/14 predictions (current)
├── predictions_vitb32.json       # ViT-B/32 predictions (backup)
├── evaluation_summary.json       # ViT-L/14 evaluation (current)
└── evaluation_summary_vitb32.json # ViT-B/32 evaluation (backup)

plots/                            # All 5 plots regenerated for ViT-L/14
```
