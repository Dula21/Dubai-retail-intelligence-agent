import json
import sys
from pathlib import Path

import faiss
import numpy as np

CATALOGUE = "data/synthetic/catalogue.json"   # best guess, change if wrong
EN_INDEX = "path/to/english_index.faiss"
AR_INDEX = "path/to/arabic_index.faiss"

# --- catalogue: field names and row count ---
p = Path(CATALOGUE)
data = json.loads(p.read_text(encoding="utf-8"))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()), "-> your list is under one of these")
else:
    print(f"Catalogue rows: {len(data)}")
    print("Field names:", list(data[0].keys()))
    print("First row:", json.dumps(data[0], ensure_ascii=False, indent=2))

# --- indices: type, metric, size, normalisation ---
for name, path in (("EN", EN_INDEX), ("AR", AR_INDEX)):
    ix = faiss.read_index(path)
    metric = "INNER PRODUCT" if ix.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
    print(f"\n{name}: type={type(ix).__name__}, vectors={ix.ntotal}, dim={ix.d}, metric={metric}")
    if hasattr(ix, "reconstruct_n"):
        sample = ix.reconstruct_n(0, min(5, ix.ntotal))
        norms = np.linalg.norm(sample, axis=1)
        print(f"{name}: norms of first 5 vectors = {np.round(norms, 3).tolist()}")