"""Gulf-dialect / code-switched / spelling-variant query test on the REAL catalogue.

Run from the repo root:   python -m eval.run_gulf_test
Reuses CATALOGUE_CSV, build_groups and the not-in-catalogue list from eval/run_baseline_v2.py.

Variants of the same product ("Summer Dress", "Summer Dress Pro") are collapsed into ONE base product,
so "top-1" means the best base product and "margin" is the gap to the next DIFFERENT product.
That is the deduplicate-before-margin rule from the project's design notes.
"""
import json
import sys
from pathlib import Path

from backend.rag.retriever import RetailRetriever
from eval.run_baseline_v2 import ABSENT, CATALOGUE_CSV, PERSIST_DIR, build_groups, mean

sys.stdout.reconfigure(encoding="utf-8")

OUT_FILE = "eval/gulf_results.json"
LABEL = "gulf_baseline_6fdc697"   # change per experiment
N = 20                            # SKU candidates fetched, collapsed to distinct base products

ABAYAS = ["Abaya Classic", "Embroidered Abaya"]
# (kind, query, [acceptable base products])  -- written as plausible Gulf phrasing; edit freely.
QUERIES = [
    # --- Gulf dialect / short Arabic ---
    ("gulf", "كندورة", ["Kandura White"]),
    ("gulf", "ثوب رجالي أبيض", ["Kandura White"]),
    ("gulf", "فستان", ["Summer Dress"]),
    ("gulf", "فستان للصيف", ["Summer Dress"]),
    ("gulf", "بنطلون جينز", ["Casual Jeans"]),
    ("gulf", "جاكيت دنيم", ["Denim Jacket"]),
    ("gulf", "ملابس سباحة للمحجبات", ["Modest Swimwear"]),
    ("gulf", "جاكيت ثقيل للبرد", ["Winter Coat"]),
    ("gulf", "تنورة طويلة", ["Maxi Skirt"]),
    ("gulf", "سماعات بلوتوث", ["Wireless Earbuds", "Gaming Headset"]),
    ("gulf", "باور بانك", ["Power Bank 20000mAh"]),
    ("gulf", "ساعة سمارت", ["Smart Watch"]),
    ("gulf", "سبيكر", ["Bluetooth Speaker"]),
    ("gulf", "كفر ايفون", ["Phone Case Premium"]),
    ("gulf", "شاحن", ["Wireless Charger"]),
    ("gulf", "كاميرا للاجتماعات", ["Webcam HD"]),
    ("gulf", "ستاند لابتوب", ["Laptop Stand"]),
    ("gulf", "هيدسيت للألعاب", ["Gaming Headset"]),
    ("gulf", "فانوس", ["Ramadan Lantern"]),
    ("gulf", "بخور وعود", ["Oud Incense Set"]),
    ("gulf", "طقم فناجين قهوة", ["Arabic Coffee Set"]),
    ("gulf", "دلة", ["Copper Dallah"]),
    ("gulf", "مصلى", ["Prayer Mat Premium"]),
    ("gulf", "مخدات للمجلس", ["Decorative Cushions"]),
    ("gulf", "شموع", ["Scented Candles Set"]),
    ("gulf", "معطر جو", ["Air Freshener Oud"]),
    ("gulf", "تمر", ["Medjool Dates 1kg"]),
    ("gulf", "زعفران", ["Saffron Premium 5g"]),
    ("gulf", "هيل", ["Cardamom Pods 100g"]),
    ("gulf", "سمن", ["Ghee Premium 500g"]),
    ("gulf", "دبس الرمان", ["Pomegranate Molasses"]),
    ("gulf", "حليب ناقة", ["Camel Milk Powder"]),
    ("gulf", "شاهي كرك", ["Karak Tea Blend"]),
    ("gulf", "عطر عود", ["Oud Perfume 50ml"]),
    ("gulf", "كريم ورد للوجه", ["Rose Face Cream"]),
    ("gulf", "كحل", ["Kohl Eyeliner"]),
    ("gulf", "حبة البركة", ["Blackseed Oil"]),
    ("gulf", "سيروم فيتامين سي", ["Vitamin C Serum"]),
    ("gulf", "هدية عطور", ["Perfume Gift Set"]),
    # --- code-switched Arabic + English ---
    ("cs", "abaya للعيد", ABAYAS),
    ("cs", "kandura أبيض", ["Kandura White"]),
    ("cs", "dress صيفي", ["Summer Dress"]),
    ("cs", "earbuds بلوتوث", ["Wireless Earbuds"]),
    ("cs", "power bank كبير", ["Power Bank 20000mAh"]),
    ("cs", "smart watch رجالي", ["Smart Watch"]),
    ("cs", "oud perfume رجالي", ["Oud Perfume 50ml"]),
    ("cs", "dates فاخر", ["Medjool Dates 1kg"]),
    ("cs", "karak شاي", ["Karak Tea Blend"]),
    ("cs", "saffron أصلي", ["Saffron Premium 5g"]),
    ("cs", "serum للوجه", ["Argan Oil Serum", "Vitamin C Serum"]),
    ("cs", "hair mask أركان", ["Hair Mask Argan"]),
    ("cs", "laptop stand للمكتب", ["Laptop Stand"]),
    ("cs", "prayer mat فاخرة", ["Prayer Mat Premium"]),
    ("cs", "gift set عطور", ["Perfume Gift Set"]),
    # --- common spellings that differ from the catalogue's spelling ---
    ("spell", "عباية", ABAYAS),
    ("spell", "عباية مطرزة", ["Embroidered Abaya"]),
    ("spell", "طحينه", ["Tahini Paste"]),
    ("spell", "قهوه عربيه", ["Arabic Coffee Blend", "Arabic Coffee Set"]),
    ("spell", "حنا", ["Henna Powder"]),
]


def main() -> None:
    rt = RetailRetriever(CATALOGUE_CSV, None, PERSIST_DIR)
    rt.initialize()
    catalogue = rt.loader.load_catalogue()
    bases, groups = build_groups(catalogue)
    base_names = {b[0] for b in bases}
    sku2base = {sku: key[1] for key, skus in groups.items() for sku in skus}

    bad = [(t, q) for _, q, ts in QUERIES for t in ts if t not in base_names]
    if bad:
        raise SystemExit(f"Unknown base product names in QUERIES (typo?): {bad}")

    def ranked(text):
        """[(base_product, best_score), ...] best first, variants collapsed."""
        res = rt.vector_store.query(text, n_results=N, min_confidence=-1.0)   # gate OFF
        seen = {}
        for r in res:
            b = sku2base.get(r["sku_id"])
            if b and b not in seen:
                seen[b] = r["similarity_score"]
        return list(seen.items())

    def margin(rk):
        return round(rk[0][1] - rk[1][1], 4) if len(rk) >= 2 else 0.0

    results = []
    for kind, text, targets in QUERIES:
        rk = ranked(text)
        results.append({"kind": kind, "query": text, "targets": targets, "ranked": rk[:3],
                        "ok1": rk[0][0] in targets, "ok3": any(b in targets for b, _ in rk[:3]),
                        "score": rk[0][1], "margin": margin(rk)})
    absent = []
    for en, ar in ABSENT:
        for text in (en, ar):
            rk = ranked(text)
            absent.append({"query": text, "ranked": rk[:3], "score": rk[0][1], "margin": margin(rk)})

    print(f"[{LABEL}] queries={len(results)} absent={len(absent)} | model={rt.embedding_mgr.config.model_name}")

    print("\n=== 1. ACCURACY by query kind (top-1 product / any of top-3 products) ===")
    for kind in ("gulf", "cs", "spell", None):
        rs = [r for r in results if kind is None or r["kind"] == kind]
        print(f"{(kind or 'ALL'):<6} n={len(rs):<3} hit@1={mean([float(r['ok1']) for r in rs])}  hit@3={mean([float(r['ok3']) for r in rs])}")

    print("\n=== 2. MISSES (top-1 product is not an accepted answer) ===")
    for r in results:
        if not r["ok1"]:
            got = " | ".join(f"{b} {s}" for b, s in r["ranked"])
            print(f"  [{r['kind']}] {r['query']}  want {r['targets']}  got: {got}")

    print("\n=== 3. CAN A THRESHOLD SEPARATE RIGHT FROM WRONG? ===")
    good = [r for r in results if r["ok1"]]
    badq = [r for r in results if not r["ok1"]] + absent
    print(f"correct: n={len(good)} mean score={mean([r['score'] for r in good])} mean margin={mean([r['margin'] for r in good])}")
    print(f"wrong+absent: n={len(badq)} mean score={mean([r['score'] for r in badq])} mean margin={mean([r['margin'] for r in badq])}")
    print("margin = gap to next DIFFERENT product (variants collapsed)")
    print("rule                    correct kept   wrong/absent caught")
    for label, key, ths in (("margin >=", "margin", (0.005, 0.01, 0.015, 0.02, 0.03)),
                            ("score  >=", "score", (0.80, 0.82, 0.84, 0.86))):
        for t in ths:
            kept = mean([float(r[key] >= t) for r in good])
            caught = mean([float(r[key] < t) for r in badq])
            print(f"{label} {t:<6}          {kept:<14} {caught}")

    Path(OUT_FILE).parent.mkdir(exist_ok=True)
    Path(OUT_FILE).write_text(json.dumps({"label": LABEL, "results": results, "absent": absent},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()