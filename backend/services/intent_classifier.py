"""
services/intent_classifier.py
------------------------------
Lightweight rule-based intent classifier and language detector.

Engineering decision: rule-based over LLM-based routing.
Using an LLM as the primary router introduces two problems at free-tier scale:
  1. Latency: adds a full Groq round-trip before the actual work starts
  2. Inconsistency: LLM classifiers can drift on edge cases without versioning

Rule-based is deterministic, instant, and testable. At SME scale (hundreds of
SKUs, dozens of daily queries) a keyword matcher is entirely sufficient.

Next architectural step at scale: replace with a fine-tuned small classifier
(DistilBERT or similar) when query volume justifies the training cost and when
Gulf Arabic dialect intent patterns need proper handling.

Field-type fallback split (per architecture principles):
  - "reorder" and "analysis" intents → factual field path → hard fallback
  - "search" and "trend" intents → narrative field path → partial synthesis allowed
"""

from __future__ import annotations

import re
import structlog

logger = structlog.get_logger(__name__)

# ── Intent keyword sets ────────────────────────────────────────────────────────

REORDER_KEYWORDS_EN = {
    "reorder", "stock", "inventory", "replenish", "run out", "runout",
    "low stock", "out of stock", "order more", "restock", "units left",
    "how many left", "running low", "need more", "shortage",
}

REORDER_KEYWORDS_AR = {
    "إعادة طلب", "مخزون", "طلب", "نفاد", "إعادة تخزين", "كمية",
    "وحدات", "منخفض", "مخزن", "احتياطي",
}

ANALYSIS_KEYWORDS_EN = {
    "underperform", "why", "analysis", "analyse", "analyze", "compare",
    "performance", "reason", "insight", "explain", "versus", "vs",
    "benchmark", "poor", "slow", "not selling", "weak",
}

ANALYSIS_KEYWORDS_AR = {
    "تحليل", "لماذا", "أداء", "مقارنة", "اتجاه", "سبب", "تفسير", "بطيء",
}

TREND_KEYWORDS_EN = {
    "trending", "popular", "demand", "forecast", "predict", "upcoming",
    "season", "ramadan", "dsf", "national day", "eid", "surge", "spike",
    "what's hot", "best sellers",
}

TREND_KEYWORDS_AR = {
    "رائج", "شعبي", "طلب", "توقع", "موسم", "رمضان", "دي إس إف",
    "اليوم الوطني", "عيد", "ارتفاع",
}


def classify_intent(query: str) -> str:
    """
    Classify query intent using keyword matching.

    Returns one of: "reorder" | "analysis" | "trend" | "search"

    Priority order: reorder > analysis > trend > search (default).
    Reorder is highest priority because it triggers the most expensive
    node path (sales + inventory + seasonality) and we want to avoid
    running that path unnecessarily.

    Args:
        query: Raw user query string (Arabic, English, or mixed)

    Returns:
        Intent label string
    """
    q_lower = query.lower().strip()

    # Check reorder signals
    if any(k in q_lower for k in REORDER_KEYWORDS_EN) or \
       any(k in q_lower for k in REORDER_KEYWORDS_AR):
        logger.debug("intent_classified", intent="reorder", query=q_lower[:50])
        return "reorder"

    # Check analysis signals
    if any(k in q_lower for k in ANALYSIS_KEYWORDS_EN) or \
       any(k in q_lower for k in ANALYSIS_KEYWORDS_AR):
        logger.debug("intent_classified", intent="analysis", query=q_lower[:50])
        return "analysis"

    # Check trend signals
    if any(k in q_lower for k in TREND_KEYWORDS_EN) or \
       any(k in q_lower for k in TREND_KEYWORDS_AR):
        logger.debug("intent_classified", intent="trend", query=q_lower[:50])
        return "trend"

    # Default: treat as product search
    logger.debug("intent_classified", intent="search", query=q_lower[:50])
    return "search"


def is_factual_intent(intent: str) -> bool:
    """
    Returns True for intents that require hard fallback (no partial answers).

    Factual intents: reorder, analysis
    Narrative intents: search, trend

    Engineering decision (field-type fallback split):
    A wrong reorder quantity is unrecoverable — a retailer acts on it
    immediately. A partial trend summary can be reviewed and is still useful.
    """
    return intent in ("reorder", "analysis")


def detect_language(query: str) -> str:
    """
    Detect query language by Arabic Unicode block character ratio.

    Arabic Unicode block: U+0600–U+06FF covers Modern Standard Arabic
    and Gulf Arabic dialects. Persian/Urdu share this block but are
    uncommon in UAE retail contexts — acceptable approximation at this scale.

    Returns: "ar" | "en" | "mixed"
    """
    if not query:
        return "en"

    arabic_chars = sum(1 for c in query if "\u0600" <= c <= "\u06ff")
    total_alpha = sum(1 for c in query if c.isalpha())

    if total_alpha == 0:
        return "en"

    ratio = arabic_chars / total_alpha

    if ratio > 0.6:
        return "ar"
    if ratio > 0.1:
        return "mixed"
    return "en"


def extract_sku_references(query: str) -> list[str]:
    """
    Extract explicit SKU references from a query.

    Pattern: SKU- or sku- followed by alphanumeric characters.
    Used by the RAG node to check whether a queried entity is present
    in retrieved chunks (margin-based routing entity check).

    Example: "why is SKU-4421 underperforming" → ["SKU-4421"]
    """
    pattern = re.compile(r"\bsku[-_]?(\w+)\b", re.IGNORECASE)
    matches = pattern.findall(query)
    return [f"SKU-{m.upper()}" for m in matches]
