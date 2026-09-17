"""
Phase 1 Test Suite
==================
Minimum 20 tests required before Phase 2 begins (per project spec).
Tests cover: data integrity, language detection, document preparation,
confidence gating logic, and retriever interface contracts.

NOTE: Vector store tests (requiring ChromaDB + sentence-transformers) are
marked with @pytest.mark.integration and skipped by default.
Run with: pytest tests/ -m integration   ← requires full dependencies
Run unit only: pytest tests/              ← no model download needed

ENGINEERING PRINCIPLE: Tests written alongside implementation, not after.
This mirrors production standards at noon/G42 where CI gates merges on tests.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unittest
from datetime import date, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# DATA GENERATION TESTS (1–8)
# These run without any ML dependencies
# ---------------------------------------------------------------------------

class TestDataGeneration(unittest.TestCase):
    """Tests for the synthetic dataset generator."""

    @classmethod
    def setUpClass(cls):
        """Load generated CSV files once for all tests in this class."""
        base = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
        
        cls.catalogue_path  = os.path.join(base, "product_catalogue.csv")
        cls.sales_path      = os.path.join(base, "sales_history.csv")
        cls.inventory_path  = os.path.join(base, "inventory_snapshot.csv")
        cls.metadata_path   = os.path.join(base, "dataset_metadata.json")

        def load_csv(path):
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))

        cls.catalogue  = load_csv(cls.catalogue_path)
        cls.sales      = load_csv(cls.sales_path)
        cls.inventory  = load_csv(cls.inventory_path)
        with open(cls.metadata_path, encoding="utf-8") as f:
            cls.metadata = json.load(f)

    # T1 — Correct SKU count
    def test_01_catalogue_has_500_skus(self):
        self.assertEqual(len(self.catalogue), 500,
                         "Catalogue must have exactly 500 SKUs")

    # T2 — All SKUs have Arabic names
    def test_02_all_skus_have_arabic_names(self):
        missing = [r["sku_id"] for r in self.catalogue if not r.get("name_ar")]
        self.assertEqual(missing, [],
                         f"SKUs missing Arabic name: {missing[:5]}")

    # T3 — All SKUs have English names
    def test_03_all_skus_have_english_names(self):
        missing = [r["sku_id"] for r in self.catalogue if not r.get("name_en")]
        self.assertEqual(missing, [],
                         f"SKUs missing English name: {missing[:5]}")

    # T4 — All prices are positive AED values
    def test_04_prices_are_positive_aed(self):
        invalid = [r["sku_id"] for r in self.catalogue
                   if float(r["price_aed"]) <= 0]
        self.assertEqual(invalid, [],
                         f"SKUs with non-positive price: {invalid[:5]}")

    # T5 — Sales history row count
    def test_05_sales_history_row_count(self):
        expected = 500 * 730  # 500 SKUs × 730 days
        self.assertEqual(len(self.sales), expected,
                         f"Expected {expected} sales rows, got {len(self.sales)}")

    # T6 — Ramadan rows have elevated event_multiplier
    def test_06_ramadan_multiplier_is_elevated(self):
        ramadan_rows  = [r for r in self.sales if r.get("is_ramadan") == "1"]
        baseline_rows = [r for r in self.sales if r.get("is_ramadan") == "0"
                         and r.get("is_dsf") == "0"]

        self.assertGreater(len(ramadan_rows), 0, "No Ramadan rows found")

        ramadan_mult  = sum(float(r["event_multiplier"]) for r in ramadan_rows)  / len(ramadan_rows)
        baseline_mult = sum(float(r["event_multiplier"]) for r in baseline_rows) / len(baseline_rows)

        self.assertGreater(ramadan_mult, baseline_mult,
                           "Ramadan multiplier should exceed baseline")
        self.assertGreater(ramadan_mult, 1.3,
                           "Ramadan average multiplier should be > 1.3")

    # T7 — DSF rows have elevated multiplier
    def test_07_dsf_multiplier_is_elevated(self):
        dsf_rows = [r for r in self.sales if r.get("is_dsf") == "1"]
        self.assertGreater(len(dsf_rows), 0, "No DSF rows found")
        avg_mult = sum(float(r["event_multiplier"]) for r in dsf_rows) / len(dsf_rows)
        self.assertGreater(avg_mult, 1.5, "DSF multiplier should exceed 1.5")

    # T8 — Inventory has both critical and healthy SKUs
    def test_08_inventory_has_diverse_statuses(self):
        statuses = {r["inventory_status"] for r in self.inventory}
        self.assertIn("critical", statuses, "No critical inventory SKUs")
        self.assertIn("healthy",  statuses, "No healthy inventory SKUs")
        self.assertIn("overstock",statuses, "No overstock inventory SKUs")


# ---------------------------------------------------------------------------
# LANGUAGE DETECTION TESTS (9–13)
# ---------------------------------------------------------------------------

class TestLanguageDetection(unittest.TestCase):
    """Tests for the bilingual language detection utility."""

    def setUp(self):
        from backend.rag.embeddings import detect_language
        self.detect = detect_language

    # T9 — English query detected
    def test_09_english_query_detected(self):
        result = self.detect("show me summer dresses under AED 200")
        self.assertEqual(result, "en")

    # T10 — Arabic query detected
    def test_10_arabic_query_detected(self):
        result = self.detect("أظهر لي الفساتين الصيفية تحت 200 درهم")
        self.assertEqual(result, "ar")

    # T11 — Mixed query detected
    def test_11_mixed_query_detected(self):
        result = self.detect("show me عباية under 300 AED")
        self.assertEqual(result, "mixed")

    # T12 — Empty / numeric input defaults to English
    def test_12_empty_input_defaults_to_english(self):
        result = self.detect("12345")
        self.assertEqual(result, "en")

    # T13 — Pure Arabic product name
    def test_13_arabic_product_name(self):
        result = self.detect("عباءة كلاسيكية")
        self.assertEqual(result, "ar")


# ---------------------------------------------------------------------------
# DOCUMENT PREPARATION TESTS (14–17)
# ---------------------------------------------------------------------------

class TestDocumentPreparation(unittest.TestCase):
    """Tests for document text construction and E5 prefix handling."""

    def setUp(self):
        from backend.rag.embeddings import (
            prepare_document_text,
            prepare_query_text,
            E5_QUERY_PREFIX,
            E5_DOCUMENT_PREFIX,
        )
        self.prepare_doc   = prepare_document_text
        self.prepare_query = prepare_query_text
        self.query_prefix  = E5_QUERY_PREFIX
        self.doc_prefix    = E5_DOCUMENT_PREFIX

    SAMPLE_PRODUCT = {
        "sku_id":           "FSH-0001",
        "name_en":          "Summer Dress",
        "name_ar":          "فستان صيفي",
        "category":         "fashion",
        "price_aed":        120.0,
        "supplier_name":    "Al Futtaim Trading",
        "supplier_lead_days": 7,
        "tags_en":          "fashion, summer dress, uae retail",
        "tags_ar":          "موضة, فستان صيفي, تجزئة الإمارات",
    }

    # T14 — Document contains English name
    def test_14_document_contains_english_name(self):
        text = self.prepare_doc(self.SAMPLE_PRODUCT)
        self.assertIn("Summer Dress", text)

    # T15 — Document contains Arabic name
    def test_15_document_contains_arabic_name(self):
        text = self.prepare_doc(self.SAMPLE_PRODUCT)
        self.assertIn("فستان صيفي", text)

    # T16 — Document contains price
    def test_16_document_contains_price(self):
        text = self.prepare_doc(self.SAMPLE_PRODUCT)
        self.assertIn("120.0", text)

    # T17 — Query text gets E5 query prefix
    def test_17_query_gets_e5_prefix(self):
        result = self.prepare_query("summer dresses")
        self.assertTrue(result.startswith(self.query_prefix),
                        f"Expected query prefix '{self.query_prefix}', got: {result[:20]}")


# ---------------------------------------------------------------------------
# CONFIDENCE GATE TESTS (18–20)
# ---------------------------------------------------------------------------

class TestConfidenceGate(unittest.TestCase):
    """Tests for the confidence threshold logic (no model needed)."""

    def setUp(self):
        from backend.rag.embeddings import EmbeddingManager, EmbeddingConfig, CONFIDENCE_THRESHOLD
        self.threshold  = CONFIDENCE_THRESHOLD
        # Use default config — model NOT loaded (no embed calls here)
        self.config     = EmbeddingConfig()
        self.mgr        = EmbeddingManager(self.config)

    # T18 — Threshold constant is 0.72
    def test_18_confidence_threshold_is_0_72(self):
        self.assertEqual(self.threshold, 0.72,
                         "Confidence threshold must be 0.72 per project spec")

    # T19 — Values above threshold pass
    def test_19_above_threshold_passes(self):
        self.assertTrue(self.mgr.is_above_confidence_threshold(0.80))
        self.assertTrue(self.mgr.is_above_confidence_threshold(0.72))
        self.assertTrue(self.mgr.is_above_confidence_threshold(0.999))

    # T20 — Values below threshold fail
    def test_20_below_threshold_fails(self):
        self.assertFalse(self.mgr.is_above_confidence_threshold(0.71))
        self.assertFalse(self.mgr.is_above_confidence_threshold(0.50))
        self.assertFalse(self.mgr.is_above_confidence_threshold(0.00))


# ---------------------------------------------------------------------------
# BONUS TESTS (21–25) — Extra coverage for Phase 2 readiness
# ---------------------------------------------------------------------------

class TestCatalogueLoader(unittest.TestCase):
    """Tests for the RetailCatalogueLoader."""

    def setUp(self):
        base = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
        from backend.rag.retriever import RetailCatalogueLoader
        self.loader = RetailCatalogueLoader(
            catalogue_path=os.path.join(base, "product_catalogue.csv"),
            sales_path=os.path.join(base, "sales_history.csv"),
        )

    # T21 — Loader returns 500 catalogue rows
    def test_21_loader_returns_500_rows(self):
        catalogue = self.loader.load_catalogue()
        self.assertEqual(len(catalogue), 500)

    # T22 — Price fields are cast to float
    def test_22_price_cast_to_float(self):
        catalogue = self.loader.load_catalogue()
        for row in catalogue[:10]:
            self.assertIsInstance(row["price_aed"], float,
                                  f"price_aed not float for {row['sku_id']}")

    # T23 — Sales summary is indexed by SKU ID
    def test_23_sales_summary_indexed_by_sku(self):
        summary = self.loader.load_sales_summary()
        self.assertGreater(len(summary), 0, "Sales summary is empty")
        # All keys should look like SKU IDs
        for key in list(summary.keys())[:5]:
            self.assertRegex(key, r"^(FSH|ELC|HOM|FD|BTY)-\d{4}$")

    # T24 — Sales summary contains expected fields
    def test_24_sales_summary_has_required_fields(self):
        summary = self.loader.load_sales_summary()
        sample  = next(iter(summary.values()))
        required = {"total_units", "total_revenue", "avg_daily_units",
                    "stockout_days", "last_30d_units"}
        self.assertTrue(required.issubset(sample.keys()),
                        f"Missing fields: {required - sample.keys()}")

    # T25 — Category distribution matches spec
    def test_25_category_distribution_correct(self):
        from backend.rag.retriever import RetailCatalogueLoader
        base = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
        loader = RetailCatalogueLoader(os.path.join(base, "product_catalogue.csv"))
        catalogue = loader.load_catalogue()
        
        from collections import Counter
        counts = Counter(row["category"] for row in catalogue)
        
        expected = {"fashion": 120, "electronics": 100,
                    "home": 100, "food": 80, "beauty": 100}
        
        for cat, expected_count in expected.items():
            self.assertEqual(counts[cat], expected_count,
                             f"Category '{cat}': expected {expected_count}, got {counts[cat]}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with verbosity to see individual test names
    unittest.main(verbosity=2)
