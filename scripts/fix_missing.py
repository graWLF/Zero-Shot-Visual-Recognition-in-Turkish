import os
import sys
import time
import requests
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CLIP-Turkish-Eval/1.0 (academic research; burakdere218@gmail.com)"}
SAVE_PER_CLASS = 20

# Alternative search terms for hard classes
TARGETS = [
    ("food_turkish", "Turkish flatbread pizza", ["pide Turkish bread", "pide food", "pide flatbread"]),
    ("food_turkish", "Turkish scrambled eggs", ["menemen Turkish", "menemen dish", "menemen eggs tomato"]),
    ("landmarks", "harbor", ["harbor port boats", "seaport harbor", "marina harbor ships", "port harbor"]),
    ("landmarks", "bridge", ["bridge architecture", "stone bridge river", "suspension bridge", "bridge structure"]),
    ("landmarks", "train station", ["train station building", "railway station", "train terminal", "railroad station"]),
]


def get_urls(search_term, limit=50):
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": search_term,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        urls = []
        for page in data.get("query", {}).get("pages", {}).values():
            for info in page.get("imageinfo", []):
                if info.get("mime") in ("image/jpeg", "image/png") and info.get("url"):
                    urls.append(info["url"])
        return urls
    except Exception as e:
        print(f"  Error: {e}")
        return []


def download_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            if "image" not in resp.headers.get("Content-Type", ""):
                return None
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img
        except Exception:
            time.sleep(2)
    return None


def fill_class(domain, cls_en, search_terms):
    out_dir = os.path.join(DATA_BASE, domain, cls_en)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    count = len(existing)
    if count >= SAVE_PER_CLASS:
        print(f"  {cls_en}: already complete ({count})")
        return

    all_urls = []
    seen = set()
    for term in search_terms:
        urls = get_urls(term, limit=50)
        time.sleep(1.0)
        for u in urls:
            if u not in seen:
                all_urls.append(u)
                seen.add(u)
        if len(all_urls) >= 60:
            break

    print(f"  {cls_en}: {len(all_urls)} candidate URLs (need {SAVE_PER_CLASS - count} more)")

    for url in all_urls:
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


def main():
    for domain, cls_en, search_terms in TARGETS:
        print(f"\n{domain}/{cls_en}")
        fill_class(domain, cls_en, search_terms)
    print("\nDone.")


if __name__ == "__main__":
    main()
