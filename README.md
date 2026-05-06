# Zero-Shot Visual Recognition in Turkish

A study on how CLIP's zero-shot image classification accuracy changes when class prompts are written in Turkish instead of English, and whether BPE tokenization fragmentation correlates with that degradation.

---

## Research Question

CLIP is trained predominantly on English text-image pairs. When we replace English prompts with Turkish equivalents — keeping the model weights and images fixed — does accuracy drop? And if so, does the drop correlate with how heavily the CLIP tokenizer fragments Turkish words into subword tokens?

---

## HOW TO RUN

```bash
python run.py
```

The pipeline will:
1. Download images for all 60 classes (**already-downloaded classes are skipped automatically**)
2. Run inference with **ViT-B/32**
3. Run inference with **ViT-L/14**
4. Evaluate both models (accuracy, McNemar tests, Bonferroni correction)
5. Generate all plots including the B/32 vs L/14 comparison


### Options

```bash
python run.py --model b32   # run ViT-B/32 only
python run.py --model l14   # run ViT-L/14 only
python run.py --model both  # default — runs both in order
```

### Running scripts individually

Each script can also be run on its own:

```bash
python scripts/collect_imagenet.py          # animals
python scripts/collect_food101.py           # food international
python scripts/collect_wikimedia.py         # food Turkish, traffic signs, landmarks
python scripts/inference.py --model b32     # inference with ViT-B/32
python scripts/inference.py --model l14     # inference with ViT-L/14
python scripts/evaluate.py                  # evaluate current predictions.json
python scripts/visualize.py                 # generate all plots
```

---

## Experimental Setup

- **Models:** CLIP ViT-B/32 and ViT-L/14 (OpenAI, via HuggingFace `transformers`)
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

### ViT-B/32 — Top-1 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | 87.0% | 45.0% | 44.5% |
| Food International | 97.5% | 39.5% | 53.0% |
| Food Turkish | 50.5% | 31.0% | 35.5% |
| Traffic Signs | 56.1% | 9.3% | 7.9% |
| Landmarks | 80.3% | 39.4% | 44.7% |

### ViT-L/14 — Top-1 Accuracy

| Domain | English | Turkish | Mixed |
|--------|---------|---------|-------|
| Animals | 95.0% | 46.0% | 47.5% |
| Food International | 100.0% | 57.5% | 73.0% |
| Food Turkish | 65.5% | 31.5% | 40.5% |
| Traffic Signs | 68.9% | 10.4% | 8.9% |
| Landmarks | 85.6% | 41.6% | 43.8% |

### Model Comparison — English→Turkish Accuracy Gap

| Domain | B/32 Gap | L/14 Gap |
|--------|----------|----------|
| Animals | −42.0 pp | −49.0 pp |
| Food International | −58.0 pp | −42.5 pp |
| Food Turkish | −19.5 pp | −34.0 pp |
| Traffic Signs | −46.8 pp | −58.5 pp |
| Landmarks | −40.9 pp | −44.0 pp |

ViT-L/14 raises English accuracy substantially but barely improves Turkish — the gap widens in 4 out of 5 domains. Scaling up model capacity does not close the language barrier.

### Statistical Significance (McNemar test, Bonferroni α = 0.0033)

English significantly outperforms Turkish in **all 5 domains** with both models. The traffic signs domain shows the most dramatic drop (56.1% → 9.3% for ViT-B/32), which also has the highest BPE fragmentation rate (1.419 tokens/word).

### BPE Fragmentation Rates

| Domain | Avg Fragmentation Rate | B/32 Accuracy Drop |
|--------|----------------------|--------------------|
| Traffic Signs | 1.419 | −46.8 pp |
| Landmarks | 1.105 | −40.9 pp |
| Food International | 1.084 | −58.0 pp |
| Animals | 1.027 | −42.0 pp |
| Food Turkish | 0.693 | −19.5 pp |

Food Turkish has the lowest fragmentation (many Turkish food words like *börek*, *pide* are short) and shows the smallest accuracy gap between conditions.

---

## Project Structure

```
.
├── run.py                        # Single entry point — runs full pipeline
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
│   ├── fix_missing.py        # Fallback collector for hard-to-find classes
│   ├── inference.py          # CLIP zero-shot inference
│   ├── evaluate.py           # Accuracy + McNemar + Bonferroni
│   └── visualize.py          # 5 result plots
├── results/
│   ├── predictions.json             # Current model predictions (1200 entries)
│   ├── evaluation_summary.json      # Current model evaluation
│   ├── predictions_vitb32.json      # ViT-B/32 backup
│   └── evaluation_summary_vitb32.json
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

## Hardware

Tested on:
- GPU: NVIDIA RTX 3050 Mobile (4 GB VRAM)
- Models: CLIP ViT-B/32 and ViT-L/14 — both ran at batch size 16 without OOM
- Inference time: ~2 minutes per model for 1,200 images
