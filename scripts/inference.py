import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompts import CLASSES, get_prompts

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
BATCH_SIZE = 16
MODEL_NAME = "openai/clip-vit-base-patch32"

DOMAINS = ["animals", "food_international", "food_turkish", "traffic_signs", "landmarks"]
CONDITIONS = ["english", "turkish", "mixed"]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor, device


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output.last_hidden_state[:, 0, :]


def encode_texts(model, processor, device, prompts):
    texts = [p["prompt"] for p in prompts]
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = _to_tensor(model.get_text_features(**inputs))
    return F.normalize(embeddings, dim=-1)


def encode_images_batch(model, processor, device, images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = _to_tensor(model.get_image_features(**inputs))
    return F.normalize(embeddings, dim=-1)


def collect_images(domain):
    domain_dir = DATA_DIR / domain
    entries = []
    for cls in CLASSES[domain]:
        cls_en = cls["en"]
        cls_tr = cls["tr"]
        cls_dir = domain_dir / cls_en
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} does not exist")
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            entries.append({
                "image_path": str(img_path.relative_to(REPO_ROOT)),
                "domain": domain,
                "true_label_en": cls_en,
                "true_label_tr": cls_tr,
            })
    return entries


def get_top_predictions(similarities, prompts, k=5):
    values, indices = similarities.topk(min(k, len(prompts)))
    labels = [prompts[i]["label"] for i in indices.tolist()]
    return labels


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, processor, device = load_model()

    print("Pre-computing text embeddings...")
    text_embeddings = {}
    label_lists = {}
    for domain in DOMAINS:
        text_embeddings[domain] = {}
        label_lists[domain] = {}
        for condition in CONDITIONS:
            prompts = get_prompts(domain, condition)
            text_embeddings[domain][condition] = encode_texts(model, processor, device, prompts)
            label_lists[domain][condition] = prompts

    print("Collecting image paths...")
    all_entries = []
    for domain in DOMAINS:
        entries = collect_images(domain)
        all_entries.extend(entries)
    print(f"Total images: {len(all_entries)}")

    results = []
    for batch_start in range(0, len(all_entries), BATCH_SIZE):
        batch = all_entries[batch_start:batch_start + BATCH_SIZE]

        images = []
        for entry in batch:
            try:
                img = Image.open(REPO_ROOT / entry["image_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  Error loading {entry['image_path']}: {e}")
                images.append(Image.new("RGB", (224, 224)))

        img_embeddings = encode_images_batch(model, processor, device, images)

        for i, entry in enumerate(batch):
            domain = entry["domain"]
            result = {
                "image_path": entry["image_path"],
                "domain": domain,
                "true_label_en": entry["true_label_en"],
                "true_label_tr": entry["true_label_tr"],
            }
            img_emb = img_embeddings[i]

            for condition in CONDITIONS:
                text_emb = text_embeddings[domain][condition]
                prompts = label_lists[domain][condition]
                sims = (img_emb @ text_emb.T)
                top5 = get_top_predictions(sims, prompts, k=5)
                top1 = top5[0]
                true_en = entry["true_label_en"]
                result[condition] = {
                    "top1": top1,
                    "top5": top5,
                    "correct_top1": top1 == true_en,
                    "correct_top5": true_en in top5,
                }

            results.append(result)

        if (batch_start // BATCH_SIZE + 1) % 10 == 0:
            done = batch_start + len(batch)
            print(f"  Processed {done}/{len(all_entries)} images")

    out_path = RESULTS_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
