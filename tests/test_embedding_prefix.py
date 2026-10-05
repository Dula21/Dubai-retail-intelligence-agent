from backend.rag.embeddings import E5_DOCUMENT_PREFIX, prepare_document_text, prepare_query_text

PRODUCT = {
    "name_en": "Summer Dress", "name_ar": "فستان صيفي", "category": "fashion",
    "tags_en": "summer", "tags_ar": "صيف", "price_aed": 150.0,
    "supplier_name": "TestCo", "supplier_lead_days": 14,
}

def test_document_text_starts_with_passage_prefix():
    assert prepare_document_text(PRODUCT).startswith(E5_DOCUMENT_PREFIX)

def test_document_prefix_appears_exactly_once():
    assert prepare_document_text(PRODUCT).count(E5_DOCUMENT_PREFIX) == 1

def test_document_text_contains_both_languages():
    text = prepare_document_text(PRODUCT)
    assert "Summer Dress" in text and "فستان صيفي" in text

def test_query_text_starts_with_query_prefix():
    assert prepare_query_text("فستان").startswith("query: ")