# Zero-Shot Visual Recognition in Turkish

A study on how CLIP's zero-shot image classification accuracy changes when class prompts are written in Turkish instead of English, and whether BPE tokenization fragmentation correlates with that degradation.

---

## Research Question

CLIP is trained predominantly on English text-image pairs. When we replace English prompts with Turkish equivalents — keeping the model weights and images fixed — does accuracy drop? And if so, does the drop correlate with how heavily the CLIP tokenizer fragments Turkish words into subword tokens?

---

## Experimental Setup

- **Model:** CLIP ViT-B/32 (OpenAI, via HuggingFace `transformers`)
- **Images:** 1,200 total — 20 images per class, collected from HuggingFace Food-101 and Wikimedia Commons
- **Domains:** 5 visual domains, 60 classes total
- **Prompt conditions:** 3

| Condition | Template | Class name |
|-----------|----------|------------|
| English | `a photo of a {}` | English |
| Turkish | `bir {} fotoğrafı` | Turkish |
| Mixed | `a photo of a {}` | Turkish |

The Mixed condition isolates the effect of the Turkish class name inside an English template.

---

## Domains & Classes

| Domain | Classes | Source |
|--------|---------|--------|
| Animals | 10 (cat, dog, elephant, lion, eagle, dolphin, horse, bear, wolf, penguin) | Wikimedia Commons |
| Food International | 10 (pizza, sushi, hamburger, pasta, fried rice, tacos, waffles, ice cream, chocolate cake, omelette) | HuggingFace Food-101 |
| Food Turkish | 10 (börek, baklava, döner, lahmacun, mercimek çorbası, menemen, köfte, pide, künefe, çorba) | Wikimedia Commons |
| Traffic Signs | 14 (stop, speed limit, yield, no parking, no entry, pedestrian crossing, roundabout, etc.) | Wikimedia Commons |
| Landmarks | 16 (Hagia Sophia, Galata Tower, Topkapi Palace, Blue Mosque, Cappadocia, Ephesus, Pamukkale, Bosphorus Bridge, etc.) | Wikimedia Commons |

---

## Results

### Top-1 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | 87.0% | 45.0% | 44.5% |
| Food International | 97.5% | 39.5% | 53.0% |
| Food Turkish | 50.5% | 31.0% | 35.5% |
| Traffic Signs | 56.1% | 9.3% | 7.9% |
| Landmarks | 80.3% | 39.4% | 44.7% |

### Statistical Significance (McNemar test, Bonferroni α = 0.0033)

English significantly outperforms Turkish in **all 5 domains**. The traffic signs domain shows the most dramatic drop (56.1% → 9.3%), which also has the highest BPE fragmentation rate (1.419 tokens/word on average for Turkish class names).

### BPE Fragmentation Rates

| Domain | Avg Fragmentation Rate |
|--------|----------------------|
| Traffic Signs | 1.419 |
| Landmarks | 1.105 |
| Food International | 1.084 |
| Animals | 1.027 |
| Food Turkish | 0.693 |

Food Turkish has the lowest fragmentation (many Turkish food words like *börek*, *pide* are short) and accordingly shows the smallest accuracy gap between English and Turkish conditions.

---

## Project Structure

```
.
├── data/
│   ├── animals/              # 10 classes × 20 images
│   ├── food_international/   # 10 classes × 20 images
│   ├── food_turkish/         # 10 classes × 20 images
│   ├── traffic_signs/        # 14 classes × 20 images
│   └── landmarks/            # 16 classes × 20 images
├── scripts/
│   ├── prompts.py            # Class definitions and prompt templates
│   ├── collect_imagenet.py   # Animals data collection (Wikimedia)
│   ├── collect_food101.py    # Food International (HuggingFace Food-101)
│   ├── collect_wikimedia.py  # Food Turkish, Traffic Signs, Landmarks
│   ├── inference.py          # CLIP zero-shot inference
│   ├── evaluate.py           # Accuracy + McNemar + Bonferroni
│   └── visualize.py          # 5 result plots
├── results/
│   ├── predictions.json      # Per-image inference results (1200 entries)
│   └── evaluation_summary.json
└── plots/
    ├── accuracy_barplot.png
    ├── fragmentation_scatter.png
    ├── confusion_matrix.png
    ├── accuracy_heatmap.png
    └── token_count_barplot.png
```

---

## Installation

**Requirements:** Python 3.10+, Conda

```bash
conda create -n CV python=3.12
conda activate CV
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets Pillow requests matplotlib seaborn scipy
```

> If you don't have a CUDA GPU, omit the `--index-url` flag and PyTorch will install the CPU version. Inference will be slower but will still work.

---

## Running

Run each step in order. Each script is independent and can be re-run safely (image collection scripts skip already-downloaded classes).

### Step 1 — Collect animal images
```bash
python scripts/collect_imagenet.py
```

### Step 2 — Collect food international images
```bash
python scripts/collect_food101.py
```

### Step 3 — Collect food Turkish, traffic signs, and landmark images
```bash
python scripts/collect_wikimedia.py
```

> Wikimedia Commons rate-limits downloads. The scripts include automatic retry logic with delays. Expect ~15–30 minutes for a full collection run.

### Step 4 — Run CLIP inference
```bash
python scripts/inference.py
```

Outputs `results/predictions.json` with top-1 and top-5 predictions for all 1,200 images across all 3 conditions.

### Step 5 — Evaluate
```bash
python scripts/evaluate.py
```

Prints accuracy tables and McNemar test results. Outputs `results/evaluation_summary.json`.

### Step 6 — Visualize
```bash
python scripts/visualize.py
```

Generates all 5 plots in `plots/`.

---

## Hardware

Tested on:
- GPU: NVIDIA RTX 3050 Mobile (4 GB VRAM)
- Model: CLIP ViT-B/32
- Inference time: ~2 minutes for 1,200 images at batch size 16
