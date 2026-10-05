import csv, collections
rows = list(csv.DictReader(open("data/synthetic/product_catalogue.csv", encoding="utf-8")))
print("columns:", list(rows[0].keys()))
print("categories:", collections.Counter(r["category"] for r in rows))
print("distinct name_en:", len({r["name_en"] for r in rows}), "of", len(rows))
for r in rows[:20] + rows[120:135]:
    print(r["sku_id"], "|", r["name_en"], "|", r["name_ar"])