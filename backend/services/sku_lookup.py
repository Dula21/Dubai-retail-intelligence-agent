"""
services/sku_lookup.py
-----------------------
Catalogue helpers used by the RAG node:

  * extract_sku_candidates()  - find catalogue-style SKU codes (FSH-0014, ELC-0123 ...) in a query
  * build_family_map()        - group variants of one product ("Summer Dress", "Summer Dress Pro")
  * build_catalogue_index()   - SKU -> catalogue row, SKU -> product family

Why this exists (measured, see eval/RESULTS.md):
  1. SKU codes are not part of the embedded text, so vector search cannot find them.
     Exact-match lookup is the right route for SKU queries (hybrid routing, principle 6).
  2. The catalogue has ~75 base products, each repeated 5-8 times as variants. Without
     collapsing variants, the top-1 vs top-3 score gap is ~0.005 for right AND wrong answers,
     so the margin signal carries no information. Collapse variants first, then compare
     distinct products.

Assumption (true for the synthetic catalogue, document it for any real one):
  a variant's name is its base product's name followed by extra words
  ("Summer Dress" -> "Summer Dress Pro"), within the same category.
  At SME scale (hundreds to low thousands of SKUs) this is O(words) per SKU.
  Next architectural step at scale: an explicit product_family column in the catalogue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 2-4 letters, hyphen or underscore, 3-5 digits:  FSH-0014, ELC-0123, SKU-4421, sku_4421
_SKU_PATTERN = re.compile(r"\b([A-Za-z]{2,4})[-_](\d{3,5})\b")


def extract_sku_candidates(query: str) -> list[str]:
    """
    Return catalogue-style SKU codes found in the query, upper-cased with a hyphen,
    in order of appearance, without duplicates.

    "how many units of fsh-0014 left" -> ["FSH-0014"]
    "USB-C hub", "T-Shirt"            -> []   (no digits)
    """
    seen: list[str] = []
    for m in _SKU_PATTERN.finditer(query or ""):
        code = f"{m.group(1).upper()}-{m.group(2)}"
        if code not in seen:
            seen.append(code)
    return seen


def build_family_map(rows: list[dict]) -> dict[str, str]:
    """
    Map every sku_id to its product family (the base product's English name).

    The family is the SHORTEST name in the same category that is a whole-word prefix of
    the SKU's own name. A SKU with no shorter prefix is its own family.
    """
    names_by_category: dict[str, set[str]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("name_en"):
            names_by_category.setdefault(str(r.get("category", "")), set()).add(str(r["name_en"]).strip())

    family: dict[str, str] = {}
    for r in rows:
        if not isinstance(r, dict) or not r.get("sku_id") or not r.get("name_en"):
            continue
        name = str(r["name_en"]).strip()
        known = names_by_category.get(str(r.get("category", "")), set())
        words = name.split()
        family[str(r["sku_id"])] = next(
            (" ".join(words[:i]) for i in range(1, len(words) + 1) if " ".join(words[:i]) in known),
            name,
        )
    return family


@dataclass(frozen=True)
class CatalogueIndex:
    by_sku: dict[str, dict] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """True only when a real catalogue was loaded."""
        return bool(self.by_sku)


def build_catalogue_index(rows: list[dict]) -> CatalogueIndex:
    good = [r for r in rows if isinstance(r, dict) and r.get("sku_id")]
    return CatalogueIndex(
        by_sku={str(r["sku_id"]).upper(): r for r in good},
        family=build_family_map(good),
    )