"""Parallel query test: Arabic vs English retrieval alignment (Insaf's experiments 1-3).

Measures, on the synthetic 500-SKU catalogue:
  1. top-5 overlap between English and Arabic/code-switched versions of each query
  2. hit@5 per query language x index (the alignment penalty)
  3. share of correct top-1 hits scoring below the 0.72 gate (Step 5 preview)
"""
import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# ---------- CONFIG: change these to match your project ----------
CATALOGUE = "data/synthetic/catalogue.json"   # list of dicts; row order == FAISS row order
SKU_FIELD = "sku"
NAME_EN_FIELD = "name_en"
EN_INDEX = "path/to/english_index.faiss"
AR_INDEX = "path/to/arabic_index.faiss"
NORMALIZE = True    # True if indices were built with normalised vectors + IndexFlatIP
K = 5
GATE = 0.72
# -----------------------------------------------------------------

# (intent, english, gulf_arabic, code_switched, regex matched against name_en to find relevant SKUs)
QUERIES_RAW = [
    # product-name style (30)
    ("name", "sports shoes", "جزمة رياضية", "جزمة sport", r"shoe|sneaker|trainer"),
    ("name", "white t-shirt", "تيشيرت أبيض", "تيشيرت white", r"t-?shirt"),
    ("name", "Samsung phone", "جوال سامسونج", "جوال Samsung", r"samsung.*(phone|galaxy)"),
    ("name", "moisturizer cream", "كريم مرطب", "كريم moisturizer", r"moisturi[sz]er"),
    ("name", "men's perfume", "عطر رجالي", "perfume رجالي", r"perfume|fragrance|cologne|oud"),
    ("name", "handbag", "شنطة يد", "شنطة handbag", r"handbag|bag"),
    ("name", "wristwatch", "ساعة يد", "ساعة watch", r"watch"),
    ("name", "sunglasses", "نظارة شمس", "نظارة sunglasses", r"sunglass"),
    ("name", "Apple laptop", "لابتوب أبل", "Apple لابتوب", r"laptop|macbook"),
    ("name", "wireless earbuds", "سماعات بلوتوث", "earbuds بلوتوث", r"earbud|earphone|airpod"),
    ("name", "orange juice", "عصير برتقال", "juice برتقال", r"orange.*juice|juice.*orange"),
    ("name", "hair shampoo", "شامبو شعر", "shampoo شعر", r"shampoo"),
    ("name", "air conditioner", "مكيف", "مكيف AC", r"air conditioner|\bAC\b"),
    ("name", "Samsung TV", "تلفزيون سامسونج", "TV سامسونج", r"samsung.*(tv|television)"),
    ("name", "office chair", "كرسي مكتب", "chair مكتب", r"office chair"),
    ("name", "washing machine", "غسالة", "غسالة washer", r"washing machine|washer"),
    ("name", "large refrigerator", "ثلاجة كبيرة", "fridge كبيرة", r"refrigerator|fridge"),
    ("name", "microwave oven", "مايكرويف", "microwave", r"microwave"),
    ("name", "Nike shoes", "جزمة نايكي", "Nike جزمة", r"nike.*(shoe|sneaker|trainer)"),
    ("name", "Adidas t-shirt", "تيشيرت أديداس", "Adidas تيشيرت", r"adidas.*t-?shirt"),
    ("name", "iPhone 15", "ايفون 15", "iPhone 15 جوال", r"iphone 15"),
    ("name", "AirPods Pro", "ايربودز برو", "AirPods برو", r"airpods"),
    ("name", "black abaya", "عباية سوداء", "abaya black", r"abaya"),
    ("name", "dates gift box", "علبة تمر هدية", "gift box تمر", r"dates"),
    ("name", "Arabic coffee", "قهوة عربية", "coffee عربية", r"arabic coffee|gahwa|coffee"),
    ("name", "basmati rice", "رز بسمتي", "rice بسمتي", r"basmati|rice"),
    ("name", "olive oil", "زيت زيتون", "olive oil زيتون", r"olive oil"),
    ("name", "vacuum cleaner", "مكنسة كهربائية", "vacuum مكنسة", r"vacuum"),
    ("name", "electric blender", "خلاط كهربائي", "blender خلاط", r"blender"),
    ("name", "phone case", "كفر جوال", "كفر phone", r"phone case|case"),
    # descriptive style, Gulf phrasing (20)
    ("desc", "summer dresses under AED 200", "أبي فساتين صيفية أقل من 200 درهم", "أبي summer dress تحت 200 درهم", r"dress"),
    ("desc", "cheap earbuds with good battery", "سماعات رخيصة بطاريتها تدوم", "earbuds رخيصة battery طويلة", r"earbud|earphone|headphone"),
    ("desc", "fridge for a big family", "ثلاجة تكفي عائلة كبيرة", "fridge للعائلة الكبيرة", r"refrigerator|fridge"),
    ("desc", "comfortable shoes for walking", "جزمة مريحة للمشي", "جزمة comfortable للمشي", r"shoe|sneaker"),
    ("desc", "perfume that lasts all day", "عطر يدوم طول اليوم", "perfume يدوم all day", r"perfume|oud|fragrance"),
    ("desc", "lightweight laptop for university", "لابتوب خفيف للجامعة", "laptop خفيف للجامعة", r"laptop"),
    ("desc", "cream for dry skin", "كريم للبشرة الجافة", "cream للـ dry skin", r"cream|moisturi"),
    ("desc", "small sofa for a small living room", "كنبة صغيرة لصالة صغيرة", "كنبة small للصالة", r"sofa|couch"),
    ("desc", "winter jacket for cold weather", "جاكيت يدفي للشتاء", "jacket شتوي", r"jacket|coat"),
    ("desc", "shampoo for hair fall", "شامبو لتساقط الشعر", "shampoo لتساقط الشعر", r"shampoo"),
    ("desc", "coffee maker for home", "مكينة قهوة للبيت", "coffee machine للبيت", r"coffee (maker|machine)"),
    ("desc", "kids school bag", "شنطة مدرسية للأطفال", "school bag للأولاد", r"school bag|backpack"),
    ("desc", "fast phone charger", "شاحن جوال سريع", "fast charger للجوال", r"charger"),
    ("desc", "organic green tea", "شاي أخضر عضوي", "green tea organic", r"green tea"),
    ("desc", "gaming mouse and keyboard", "ماوس وكيبورد للألعاب", "gaming mouse و keyboard", r"mouse|keyboard"),
    ("desc", "evening dress for a wedding", "فستان سهرة للعرس", "فستان evening للعرس", r"dress|gown"),
    ("desc", "non-stick frying pan", "مقلاة ما تلصق", "مقلاة non-stick", r"frying pan|\bpan\b"),
    ("desc", "air freshener for the car", "معطر للسيارة", "air freshener للسيارة", r"freshener"),
    ("desc", "portable bluetooth speaker", "سماعة بلوتوث محمولة", "speaker بلوتوث portable", r"speaker"),
    ("desc", "baby diapers size 4", "حفايظ أطفال مقاس 4", "diapers مقاس 4", r"diaper"),
]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def search(model, index, text, skus):
    """Return [(sku, score), ...] for one query against one FAISS index."""
    vec = model.encode([text], normalize_embeddings=NORMALIZE).astype("float32")
    scores, idx = index.search(vec, K)
    return [(skus[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]


def main():
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    catalogue = json.loads(Path(CATALOGUE).read_text(encoding="utf-8"))
    skus = [c[SKU_FIELD] for c in catalogue]
    indices = {"en_idx": faiss.read_index(EN_INDEX), "ar_idx": faiss.read_index(AR_INDEX)}
    for name, ix in indices.items():
        assert ix.ntotal == len(skus), f"{name}: {ix.ntotal} vectors vs {len(skus)} catalogue rows"
        metric = "inner product" if ix.metric_type == faiss.METRIC_INNER_PRODUCT else "L2 (scores are distances!)"
        print(f"{name}: {ix.ntotal} vectors, metric = {metric}")

    rows = []
    for i, (intent, en, ar, cs, kw) in enumerate(QUERIES_RAW, 1):
        relevant = {c[SKU_FIELD] for c in catalogue if re.search(kw, c[NAME_EN_FIELD], re.I)}
        row = {"id": f"q{i:02d}", "intent": intent, "relevant_n": len(relevant), "relevant": relevant, "hits": {}}
        for variant, text in (("en", en), ("ar", ar), ("cs", cs)):
            for iname, index in indices.items():
                row["hits"][(variant, iname)] = search(model, index, text, skus)
        rows.append(row)

    scored = [r for r in rows if r["relevant_n"] > 0]
    empty = [r["id"] for r in rows if r["relevant_n"] == 0]
    groups = {
        "all": rows,
        "name": [r for r in rows if r["intent"] == "name"],
        "desc": [r for r in rows if r["intent"] == "desc"],
    }
    scored_groups = {k: [r for r in v if r["relevant_n"] > 0] for k, v in groups.items()}

    print(f"\nQueries: {len(rows)} | with ground truth: {len(scored)} | no matching SKU in catalogue: {empty}")

    print("\n=== 1. TOP-5 OVERLAP vs the English query, same index ===")
    for iname in indices:
        for v in ("ar", "cs"):
            out = {g: mean([len({s for s, _ in r["hits"][(v, iname)]} & {s for s, _ in r["hits"][("en", iname)]}) / K
                            for r in rs]) for g, rs in groups.items()}
            print(f"{v:>2} vs en on {iname}: {out}")

    print("\n=== 2. HIT@5 (any relevant SKU in top 5) ===")
    for iname in indices:
        for v in ("en", "ar", "cs"):
            out = {g: mean([1.0 if {s for s, _ in r["hits"][(v, iname)]} & r["relevant"] else 0.0
                            for r in rs]) for g, rs in scored_groups.items()}
            print(f"{v:>2} query on {iname}: {out}")

    print(f"\n=== 3. CORRECT top-1 hits scoring below the {GATE} gate ===")
    for iname in indices:
        for v in ("en", "ar", "cs"):
            correct = [r["hits"][(v, iname)][0][1] for r in scored
                       if r["hits"][(v, iname)] and r["hits"][(v, iname)][0][0] in r["relevant"]]
            below = mean([1.0 if s < GATE else 0.0 for s in correct])
            print(f"{v:>2} on {iname}: correct top-1 = {len(correct)}, mean score = {mean(correct)}, share below gate = {below}")

    serialisable = [{**r, "relevant": sorted(r["relevant"]),
                     "hits": {f"{v}@{n}": h for (v, n), h in r["hits"].items()}} for r in rows]
    Path("eval").mkdir(exist_ok=True)
    Path("eval/parallel_query_results.json").write_text(
        json.dumps(serialisable, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved eval/parallel_query_results.json")


if __name__ == "__main__":
    main()