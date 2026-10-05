import csv
from collections import defaultdict
rows = list(csv.DictReader(open("data/synthetic/product_catalogue.csv", encoding="utf-8")))
seen = defaultdict(int)
for r in rows:
    seen[r["category"]] += 1
    if seen[r["category"]] <= 15:
        print(r["category"], "|", r["name_en"], "|", r["name_ar"])