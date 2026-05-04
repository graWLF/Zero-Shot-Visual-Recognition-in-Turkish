import os
import sys
import time
import requests
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompts import CLASSES

SAVE_PER_CLASS = 20
DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CLIP-Turkish-Eval/1.0 (academic research; burakdere218@gmail.com)"}

DOMAINS = ["food_turkish", "traffic_signs", "landmarks"]


def get_image_urls(search_term, limit=80):
    urls = []

    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": search_term,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        for page in data.get("query", {}).get("pages", {}).values():
            for info in page.get("imageinfo", []):
                mime = info.get("mime", "")
                url = info.get("url", "")
                if mime in ("image/jpeg", "image/png") and url:
                    urls.append(url)
    except Exception as e:
        print(f"  Search error for '{search_term}': {e}")
    time.sleep(0.5)

    return urls


def download_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                return None
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img
        except Exception:
            time.sleep(2)
    return None


def download_class(cls_en, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    if len(existing) >= SAVE_PER_CLASS:
        print(f"  {cls_en}: already has {len(existing)} images, skipping")
        return len(existing)

    urls = get_image_urls(cls_en, limit=80)

    if len(urls) < SAVE_PER_CLASS:
        extra = get_image_urls(f"{cls_en} photo", limit=50)
        seen = set(urls)
        for u in extra:
            if u not in seen:
                urls.append(u)
                seen.add(u)

    print(f"  {cls_en}: {len(urls)} candidate URLs")

    count = 0
    for url in urls:
        if count >= SAVE_PER_CLASS:
            break
        img = download_with_retry(url)
        if img is None:
            continue
        out_path = os.path.join(out_dir, f"{count+1:03d}.jpg")
        img.save(out_path, "JPEG")
        count += 1
        time.sleep(1.5)

    if count < SAVE_PER_CLASS:
        print(f"WARNING: {cls_en} only has {count}/{SAVE_PER_CLASS} images")
    else:
        print(f"  {cls_en}: {count} images saved")
    return count


def main():
    for domain in DOMAINS:
        print(f"\n=== {domain} ===")
        out_base = os.path.join(DATA_BASE, domain)
        os.makedirs(out_base, exist_ok=True)
        for cls in CLASSES[domain]:
            cls_en = cls["en"]
            out_dir = os.path.join(out_base, cls_en)
            download_class(cls_en, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
