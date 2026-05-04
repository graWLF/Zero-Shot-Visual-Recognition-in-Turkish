CLASSES = {
    "animals": [
        {"en": "cat",      "tr": "kedi"},
        {"en": "dog",      "tr": "köpek"},
        {"en": "elephant", "tr": "fil"},
        {"en": "lion",     "tr": "aslan"},
        {"en": "eagle",    "tr": "kartal"},
        {"en": "dolphin",  "tr": "yunus"},
        {"en": "horse",    "tr": "at"},
        {"en": "bear",     "tr": "ayı"},
        {"en": "wolf",     "tr": "kurt"},
        {"en": "penguin",  "tr": "penguen"},
    ],
    "food_international": [
        {"en": "pizza",           "tr": "pizza"},
        {"en": "sushi",           "tr": "suşi"},
        {"en": "hamburger",       "tr": "hamburger"},
        {"en": "pasta",           "tr": "makarna"},
        {"en": "fried rice",      "tr": "kızarmış pirinç"},
        {"en": "tacos",           "tr": "tako"},
        {"en": "waffles",         "tr": "waffle"},
        {"en": "ice cream",       "tr": "dondurma"},
        {"en": "chocolate cake",  "tr": "çikolatalı kek"},
        {"en": "omelette",        "tr": "omlet"},
    ],
    "food_turkish": [
        {"en": "Turkish pastry",          "tr": "börek"},
        {"en": "baklava",                 "tr": "baklava"},
        {"en": "Turkish kebab",           "tr": "döner"},
        {"en": "Turkish flatbread",       "tr": "lahmacun"},
        {"en": "lentil soup",             "tr": "mercimek çorbası"},
        {"en": "Turkish scrambled eggs",  "tr": "menemen"},
        {"en": "Turkish meatballs",       "tr": "köfte"},
        {"en": "Turkish flatbread pizza", "tr": "pide"},
        {"en": "Turkish cheese dessert",  "tr": "künefe"},
        {"en": "soup",                    "tr": "çorba"},
    ],
    "traffic_signs": [
        {"en": "stop sign",                "tr": "dur işareti"},
        {"en": "speed limit sign",         "tr": "hız sınırı işareti"},
        {"en": "yield sign",               "tr": "yol ver işareti"},
        {"en": "no parking sign",          "tr": "park yasağı işareti"},
        {"en": "no entry sign",            "tr": "giriş yasağı işareti"},
        {"en": "pedestrian crossing sign", "tr": "yaya geçidi işareti"},
        {"en": "roundabout sign",          "tr": "dönel kavşak işareti"},
        {"en": "city entrance sign",       "tr": "şehir girişi işareti"},
        {"en": "motorway sign",            "tr": "otoyol işareti"},
        {"en": "rest area sign",           "tr": "mola alanı işareti"},
        {"en": "rough road sign",          "tr": "bozuk yol işareti"},
        {"en": "school zone sign",         "tr": "okul bölgesi işareti"},
        {"en": "tunnel sign",              "tr": "tünel işareti"},
        {"en": "bridge sign",              "tr": "köprü işareti"},
    ],
    "landmarks": [
        {"en": "Hagia Sophia",              "tr": "Ayasofya"},
        {"en": "Galata Tower",              "tr": "Galata Kulesi"},
        {"en": "Topkapi Palace",            "tr": "Topkapı Sarayı"},
        {"en": "Blue Mosque",               "tr": "Sultanahmet Camii"},
        {"en": "Cappadocia fairy chimneys", "tr": "Kapadokya peri bacaları"},
        {"en": "Ephesus ancient city",      "tr": "Efes antik kenti"},
        {"en": "Pamukkale travertines",     "tr": "Pamukkale travertenleri"},
        {"en": "Bosphorus Bridge",          "tr": "Boğaz Köprüsü"},
        {"en": "skyscraper",                "tr": "gökdelen"},
        {"en": "city square",               "tr": "şehir meydanı"},
        {"en": "subway station",            "tr": "metro istasyonu"},
        {"en": "shopping mall",             "tr": "alışveriş merkezi"},
        {"en": "city park",                 "tr": "şehir parkı"},
        {"en": "harbor",                    "tr": "liman"},
        {"en": "bridge",                    "tr": "köprü"},
        {"en": "train station",             "tr": "tren garı"},
    ],
}

TEMPLATES = {
    "english": "a photo of a {}",
    "turkish": "bir {} fotoğrafı",
    "mixed":   "a photo of a {}",
}


def get_prompts(domain, condition):
    classes = CLASSES[domain]
    template = TEMPLATES[condition]
    results = []
    for cls in classes:
        name = cls["tr"] if condition in ("turkish", "mixed") else cls["en"]
        results.append({"label": cls["en"], "prompt": template.format(name)})
    return results
