"""
services/event_aliases.py
--------------------------
Pure data + matching: which words in a query name which UAE retail calendar event, and when a
query is an EVENT-PLANNING question ("what should I stock more of before Diwali").

No imports from the agent package, so the intent classifier (services) and the seasonality /
candidates nodes (agents) can both use it without a circular import.

Engineering decision: planning is its own intent, not a flavour of "reorder".
  A "reorder" query is a lookup about specific products and is a FACTUAL field: it goes through
  retrieval and the hard fallback. "What should I stock before Diwali" has no single product to
  look up; any similarity ranking over near-identical variants is a coin flip (top-1 and top-3 sit
  within 0.003), so gating it on retrieval margin returns "insufficient data" to a question the
  calendar can answer. Planning therefore starts from the calendar, not from vector search.
"""

from __future__ import annotations

import re

# Query phrases (English + Arabic) -> exact calendar event names (data/synthetic/uae_retail_calendar.csv).
# Matched on word boundaries so "eid" does not match "side". A test checks every name exists in the calendar.
EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "Ramadan": ("ramadan", "ramadhan", "رمضان"),
    "Eid al-Fitr": ("eid al-fitr", "eid al fitr", "eid ul fitr", "عيد الفطر"),
    "Eid al-Adha": ("eid al-adha", "eid al adha", "eid ul adha", "عيد الأضحى", "عيد الاضحى"),
    "Diwali": ("diwali", "deepavali", "ديوالي", "دوالي"),
    "Holi": ("holi", "هولي"),
    "Navratri-Dussehra": ("navratri", "dussehra", "dasara", "نافراتري"),
    "UAE National Day": ("national day", "اليوم الوطني", "عيد الاتحاد"),
    "Christmas": ("christmas", "xmas", "كريسماس", "الكريسماس", "عيد الميلاد"),
    "New Year / Year-End Sale": ("new year", "year-end", "year end", "رأس السنة"),
    "Valentine's Day": ("valentine", "valentines", "فالنتين", "عيد الحب"),
    "Mother's Day": ("mother's day", "mothers day", "mother day", "عيد الأم", "عيد الام"),
    "Back to School": ("back to school", "back-to-school", "العودة للمدارس", "العودة إلى المدارس"),
    "Summer Season": ("summer", "الصيف"),
    "Winter Season": ("winter", "الشتاء"),
    "Singles Day 11.11": ("singles day", "11.11", "يوم العزاب"),
    "White Friday": ("white friday", "black friday", "الجمعة البيضاء", "الجمعة السوداء"),
    "Dubai Shopping Festival": ("dsf", "shopping festival", "مهرجان دبي للتسوق"),
}
# Only used when no specific phrase matched: "eid" alone means both Eids.
GENERIC_ALIASES: dict[str, tuple[str, ...]] = {
    "eid": ("Eid al-Fitr", "Eid al-Adha"),
    "العيد": ("Eid al-Fitr", "Eid al-Adha"),
}

# Words that turn "an event was mentioned" into "the retailer is planning stock for it".
PLANNING_WORDS = (
    "stock", "restock", "reorder", "order", "inventory", "prepare", "preparing", "plan", "planning",
    "buy", "purchase", "procure", "supply", "stock up", "need more", "run out",
    "اخزن", "أخزن", "نخزن", "تخزين", "مخزون", "اطلب", "أطلب", "نطلب", "طلبية",
    "استعداد", "أستعد", "استعد", "اجهز", "أجهز", "اشتري", "أشتري", "نشتري", "أحتاج", "احتاج",
)
# Forward-looking planning questions that name no event ("what should I stock up on for the season").
OPEN_PLANNING_PHRASES = (
    "what should i stock", "what to stock", "what should i restock", "what to restock",
    "stock up", "stock more of", "what should i order", "what to order",
    "ماذا أخزن", "ماذا اخزن", "ما الذي يجب أن أخزن", "ما الذي أخزن", "ايش اخزن", "شنو اخزن",
)
FORWARD_WORDS = ("before", "ahead of", "upcoming", "next", "coming", "season", "holiday", "festive",
                 "قبل", "القادم", "القادمة", "موسم", "الموسم", "العيد")

# A catalogue SKU in the query (FSH-0001, HOM-0259, FD-0359) means a product lookup, not planning.
_SKU_CODE = re.compile(r"\b[A-Za-z]{2,4}-\d{3,5}\b")


_ARABIC = re.compile(r"[\u0600-\u06FF]")
# Arabic glues one-letter prepositions onto the next word: لعيد الفطر = "for Eid al-Fitr", برمضان = "in Ramadan".
_AR_PREFIX = "(?:[لبوفك])?"


def _phrase_in(phrase: str, text: str) -> bool:
    prefix = _AR_PREFIX if _ARABIC.search(phrase) else ""
    return re.search(rf"(?<!\w){prefix}{re.escape(phrase)}(?!\w)", text) is not None


def named_event_names(query: str) -> set[str]:
    """Calendar event names mentioned in the query (specific phrases first, generic 'eid' as a fallback)."""
    q = (query or "").lower()
    specific = {name for name, phrases in EVENT_ALIASES.items() if any(_phrase_in(p, q) for p in phrases)}
    if specific:
        return specific
    generic: set[str] = set()
    for phrase, names in GENERIC_ALIASES.items():
        if _phrase_in(phrase, q):
            generic.update(names)
    return generic


def is_event_planning_query(query: str) -> bool:
    """
    True for 'what should I stock before <event>' style questions.

      * a named event AND a planning word                      -> True
      * an open planning phrase AND a forward-looking word     -> True (event chosen from the calendar)
      * a catalogue SKU code in the query                      -> False (that is a product lookup)
    """
    q = (query or "").lower().strip()
    if not q or _SKU_CODE.search(query or ""):
        return False
    planning = any(w in q for w in PLANNING_WORDS)
    if named_event_names(q) and planning:
        return True
    return (any(p in q for p in OPEN_PLANNING_PHRASES)
            and any(w in q for w in FORWARD_WORDS))