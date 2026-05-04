import os
import sys
import time
import requests
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompts import CLASSES

SAVE_PER_CLASS = 20
OUT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "animals")
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CLIP-Turkish-Eval/1.0 (academic research; burakdere218@gmail.com)"}

CATEGORY_MAP = {
    "cat":      "Cats",
    "dog":      "Dogs",
    "elephant": "Elephants",
    "lion":     "Lions",
    "eagle":    "Eagles",
    "dolphin":  "Dolphins",
    "horse":    "Horses",
    "bear":     "Bears",
    "wolf":     "Wolves",
    "penguin":  "Penguins",
}


def get_image_urls(category, search_term, limit=80):
    urls = []

    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": limit,
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
        print(f"  Category API error: {e}")
    time.sleep(1.0)

    if len(urls) < limit:
        params2 = {
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
            resp = requests.get(API_URL, params=params2, headers=HEADERS, timeout=15)
            data = resp.json()
            seen = set(urls)
            for page in data.get("query", {}).get("pages", {}).values():
                for info in page.get("imageinfo", []):
                    mime = info.get("mime", "")
                    url = info.get("url", "")
                    if mime in ("image/jpeg", "image/png") and url and url not in seen:
                        urls.append(url)
                        seen.add(url)
        except Exception as e:
            print(f"  Search API error: {e}")
        time.sleep(1.0)

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


def download_images(cls_en, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    if len(existing) >= SAVE_PER_CLASS:
        print(f"  {cls_en}: already has {len(existing)} images, skipping")
        return len(existing)

    category = CATEGORY_MAP.get(cls_en, cls_en.capitalize() + "s")
    urls = get_image_urls(category, cls_en, limit=80)
    print(f"  {cls_en}: found {len(urls)} candidate URLs")

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
    os.makedirs(OUT_BASE, exist_ok=True)
    for cls in CLASSES["animals"]:
        cls_en = cls["en"]
        out_dir = os.path.join(OUT_BASE, cls_en)
        download_images(cls_en, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
