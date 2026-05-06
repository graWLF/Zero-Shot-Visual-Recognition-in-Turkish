import os
import sys
from datasets import load_dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompts import CLASSES

SAVE_PER_CLASS = 20
OUT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "food_international")

# food101 label names use underscores
FOOD101_LABEL_MAP = {
    "pizza":          ["pizza"],
    "sushi":          ["sushi"],
    "hamburger":      ["hamburger"],
    "pasta":          ["spaghetti_bolognese", "spaghetti_carbonara", "fettuccine_alfredo", "pasta"],
    "fried rice":     ["fried_rice"],
    "tacos":          ["tacos"],
    "waffles":        ["waffles"],
    "ice cream":      ["ice_cream"],
    "chocolate cake": ["chocolate_cake"],
    "omelette":       ["omelette"],
}


def main():
    print("Loading food101 dataset...")
    ds = load_dataset("food101", split="train", streaming=True)

    label_names = ds.features["label"].names
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    class_to_indices = {}
    for cls_en, food101_labels in FOOD101_LABEL_MAP.items():
        indices = set()
        for flabel in food101_labels:
            for name, idx in label_to_idx.items():
                if flabel.lower() in name.lower():
                    indices.add(idx)
        class_to_indices[cls_en] = indices
        matched = [label_names[i] for i in sorted(indices)]
        print(f"  {cls_en}: matched {matched}")

    os.makedirs(OUT_BASE, exist_ok=True)
    for cls in CLASSES["food_international"]:
        os.makedirs(os.path.join(OUT_BASE, cls["en"]), exist_ok=True)

    # Skip classes that already have enough images
    counts = {}
    done = set()
    for cls in CLASSES["food_international"]:
        cls_en = cls["en"]
        existing = len([f for f in os.listdir(os.path.join(OUT_BASE, cls_en)) if f.endswith(".jpg")])
        counts[cls_en] = existing
        if existing >= SAVE_PER_CLASS:
            print(f"  {cls_en}: already has {existing} images, skipping")
            done.add(cls_en)

    if len(done) == len(FOOD101_LABEL_MAP):
        print("All food_international classes already complete.")
        return

    print("\nStreaming dataset...")
    for example in ds:
        if len(done) == len(FOOD101_LABEL_MAP):
            break

        label_idx = example["label"]
        for cls_en, indices in class_to_indices.items():
            if cls_en in done:
                continue
            if label_idx in indices and counts[cls_en] < SAVE_PER_CLASS:
                img = example["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                out_path = os.path.join(OUT_BASE, cls_en, f"{counts[cls_en]+1:03d}.jpg")
                img.save(out_path, "JPEG")
                counts[cls_en] += 1
                if counts[cls_en] == SAVE_PER_CLASS:
                    print(f"  {cls_en}: {SAVE_PER_CLASS} images saved")
                    done.add(cls_en)

    for cls_en, count in counts.items():
        if count < SAVE_PER_CLASS:
            print(f"WARNING: {cls_en} only has {count}/{SAVE_PER_CLASS} images")

    print("\nDone.")


if __name__ == "__main__":
    main()
