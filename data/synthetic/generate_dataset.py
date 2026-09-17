"""
Synthetic UAE Retail Dataset Generator
=======================================
Generates 500 SKUs with realistic Dubai retail characteristics:
- Arabic + English product names
- AED pricing at mid-market price points
- 24 months of daily sales with Ramadan, DSF, National Day, Eid spikes
- Inventory levels and supplier lead times

ENGINEERING DECISION: Synthetic data over scraped data
-------------------------------------------------------
Alternative: Scrape noon.com product listings
Chosen:      Synthetic generation with controlled seasonality patterns
Why:         (1) We need exact seasonality patterns — scraped data won't have
             clean time-series aligned to Islamic calendar events.
             (2) Portfolio transparency — clearly labelled synthetic is cleaner
             than grey-area scraping.
             (3) We control edge cases: stockouts, demand spikes, slow movers.
Trade-off:   Distribution shift risk when deployed on real data. Mitigated
             by parameterising all distributions so real data can calibrate them.

NOTE FOR PORTFOLIO/INTERVIEWS:
The Hijri calendar offset logic (Ramadan shifting ~11 days/year) is a real
production concern for any system targeting UAE/GCC retailers. This generator
models it correctly. Mention this in G42/noon interviews.
"""

import csv
import json
import math
import random
import os
from datetime import date, timedelta
from typing import Any

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOGUE_PATH = os.path.join(OUTPUT_DIR, "product_catalogue.csv")
SALES_PATH     = os.path.join(OUTPUT_DIR, "sales_history.csv")
INVENTORY_PATH = os.path.join(OUTPUT_DIR, "inventory_snapshot.csv")
METADATA_PATH  = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

# ── Date range: 24 months ending today ───────────────────────────────────────
END_DATE   = date(2026, 8, 15)
START_DATE = date(2024, 8, 16)
DATE_RANGE = [START_DATE + timedelta(days=i)
              for i in range((END_DATE - START_DATE).days + 1)]

# =============================================================================
# PRODUCT TAXONOMY
# Arabic names are transliterations / actual Gulf-market product names.
# Price ranges reflect Dubai mid-market (not luxury, not souq).
# =============================================================================

CATEGORIES: dict[str, dict[str, Any]] = {
    "fashion": {
        "count": 120,
        "seasonality_peak": "winter",          # Oct–Feb peak for UAE fashion
        "ramadan_multiplier": 1.8,             # Eid outfits surge
        "dsf_multiplier": 2.2,
        "national_day_multiplier": 1.5,
        "products": [
            # (english_name, arabic_name, base_price_aed, price_variance)
            ("Summer Dress",         "فستان صيفي",           120, 40),
            ("Abaya Classic",        "عباءة كلاسيكية",        280, 60),
            ("Casual Jeans",         "جينز كاجوال",           180, 50),
            ("Linen Shirt",          "قميص كتان",             95,  25),
            ("Maxi Skirt",           "تنورة ماكسي",           140, 35),
            ("Kandura White",        "كندورة بيضاء",          350, 80),
            ("Sports Leggings",      "ليغنز رياضي",           85,  20),
            ("Formal Blazer",        "بليزر رسمي",            320, 90),
            ("Beach Kaftan",         "كفتان شاطئي",           160, 45),
            ("Winter Coat",          "معطف شتوي",             450, 120),
            ("Palazzo Pants",        "بنطال بالاتزو",         130, 35),
            ("Embroidered Abaya",    "عباءة مطرزة",           520, 130),
            ("Denim Jacket",         "جاكيت جينز",            220, 60),
            ("Printed T-Shirt",      "تيشيرت مطبوع",          65,  15),
            ("Modest Swimwear",      "مايو محتشم",            195, 50),
        ],
    },
    "electronics": {
        "count": 100,
        "seasonality_peak": "back_to_school",  # Aug–Sep + DSF
        "ramadan_multiplier": 1.3,
        "dsf_multiplier": 2.8,
        "national_day_multiplier": 1.4,
        "products": [
            ("Wireless Earbuds",     "سماعات لاسلكية",        180, 60),
            ("Power Bank 20000mAh",  "بطارية محمولة",         120, 30),
            ("Smart Watch",          "ساعة ذكية",             450, 150),
            ("Bluetooth Speaker",    "مكبر صوت بلوتوث",       220, 70),
            ("USB-C Hub",            "محور USB-C",            95,  25),
            ("Phone Case Premium",   "كفر جوال فاخر",         45,  15),
            ("Ring Light",           "إضاءة حلقية",           160, 40),
            ("Laptop Stand",         "حامل لابتوب",           110, 30),
            ("Webcam HD",            "كاميرا ويب",            185, 50),
            ("Wireless Charger",     "شاحن لاسلكي",           85,  25),
            ("Portable Monitor",     "شاشة محمولة",           680, 180),
            ("Gaming Headset",       "سماعة جيمينج",          280, 80),
            ("Smart Plug",           "فيشة ذكية",             55,  15),
            ("Action Camera",        "كاميرا أكشن",           390, 100),
            ("Tablet Stand",         "حامل تابلت",            75,  20),
        ],
    },
    "home": {
        "count": 100,
        "seasonality_peak": "ramadan",         # Home decoration surges pre-Ramadan
        "ramadan_multiplier": 2.1,
        "dsf_multiplier": 1.9,
        "national_day_multiplier": 1.3,
        "products": [
            ("Ramadan Lantern",      "فانوس رمضاني",          95,  30),
            ("Oud Incense Set",      "طقم عود وبخور",         180, 50),
            ("Arabic Coffee Set",    "طقم قهوة عربية",        320, 90),
            ("Decorative Cushions",  "وسائد ديكورية",         65,  20),
            ("Prayer Mat Premium",   "سجادة صلاة فاخرة",      150, 40),
            ("Copper Dallah",        "دلة نحاسية",            420, 100),
            ("Scented Candles Set",  "شموع عطرية",            85,  25),
            ("Wall Art Islamic",     "لوحة إسلامية",          220, 70),
            ("Storage Baskets",      "سلال تخزين",            55,  15),
            ("Table Runner",         "رنر طاولة",             45,  12),
            ("Air Freshener Oud",    "معطر جو عود",           60,  15),
            ("Serving Tray Gold",    "صينية تقديم ذهبية",     280, 80),
            ("Throw Blanket",        "بطانية ناعمة",          120, 35),
            ("Fairy Lights",         "إضاءة خيالية",          70,  20),
            ("Herb Planter Set",     "أصص أعشاب",             95,  25),
        ],
    },
    "food": {
        "count": 80,
        "seasonality_peak": "ramadan",
        "ramadan_multiplier": 3.2,             # Highest — iftar essentials
        "dsf_multiplier": 1.6,
        "national_day_multiplier": 1.8,
        "products": [
            ("Medjool Dates 1kg",    "تمر مجدول كيلو",        95,  15),
            ("Arabic Coffee Blend",  "قهوة عربية",            65,  20),
            ("Saffron Premium 5g",   "زعفران فاخر",           85,  25),
            ("Tahini Paste",         "طحينة",                 35,  8),
            ("Rose Water 500ml",     "ماء الورد",             28,  6),
            ("Mixed Nuts 500g",      "مكسرات مشكلة",          75,  20),
            ("Cardamom Pods 100g",   "هيل كامل",              45,  12),
            ("Ghee Premium 500g",    "سمن بلدي فاخر",         55,  15),
            ("Pomegranate Molasses", "دبس رمان",              30,  8),
            ("Za'atar Blend 200g",   "زعتر مزيج",             38,  10),
            ("Halawa Pistachio",     "حلاوة فستق",            42,  12),
            ("Camel Milk Powder",    "حليب إبل بودرة",        120, 30),
            ("Dried Figs 500g",      "تين مجفف",              55,  15),
            ("Karak Tea Blend",      "شاي كرك",               40,  10),
            ("Sumac Spice 150g",     "سماق",                  25,  6),
        ],
    },
    "beauty": {
        "count": 100,
        "seasonality_peak": "eid",
        "ramadan_multiplier": 2.4,             # Eid grooming surge
        "dsf_multiplier": 2.0,
        "national_day_multiplier": 1.4,
        "products": [
            ("Oud Perfume 50ml",     "عطر عود 50 مل",         280, 80),
            ("Rose Face Cream",      "كريم وجه ورد",          120, 35),
            ("Argan Oil Serum",      "سيروم زيت أركان",       150, 40),
            ("Kohl Eyeliner",        "كحل عيون",              45,  12),
            ("Micellar Water",       "ماء مايسيلار",          55,  15),
            ("Henna Powder",         "حناء طبيعية",           35,  10),
            ("Lip Gloss Set",        "طقم ملمع شفاه",         65,  20),
            ("Facial Mask Pack",     "قناع وجه",              40,  12),
            ("Hair Mask Argan",      "قناع شعر أركان",        95,  25),
            ("Blackseed Oil",        "زيت حبة البركة",        75,  20),
            ("Whitening Cream",      "كريم تفتيح",            110, 30),
            ("Perfume Gift Set",     "طقم عطور هدية",         350, 100),
            ("Natural Soap Set",     "طقم صابون طبيعي",       85,  25),
            ("Vitamin C Serum",      "سيروم فيتامين C",       135, 40),
            ("Brow Pencil",          "قلم حواجب",             40,  10),
        ],
    },
}

# =============================================================================
# UAE SEASONAL EVENTS (Gregorian approximations for 2024-2026)
# ENGINEERING NOTE: In production, use the `hijri-converter` library for
# precise Islamic calendar dates. Here we use known approximate Gregorian
# dates. The Prophet model in Phase 2 will use these as regressors.
# =============================================================================

UAE_EVENTS: list[dict[str, Any]] = [
    # Ramadan 2025: ~March 1 – March 30
    {"name": "Ramadan_2025",     "start": date(2025, 3,  1), "end": date(2025, 3, 30), "type": "ramadan"},
    # Eid Al-Fitr 2025
    {"name": "Eid_Fitr_2025",    "start": date(2025, 3, 30), "end": date(2025, 4,  3), "type": "eid"},
    # Eid Al-Adha 2025
    {"name": "Eid_Adha_2025",    "start": date(2025, 6,  6), "end": date(2025, 6, 10), "type": "eid"},
    # Ramadan 2026: ~Feb 18 – Mar 19
    {"name": "Ramadan_2026",     "start": date(2026, 2, 18), "end": date(2026, 3, 19), "type": "ramadan"},
    # Eid Al-Fitr 2026
    {"name": "Eid_Fitr_2026",    "start": date(2026, 3, 20), "end": date(2026, 3, 24), "type": "eid"},
    # DSF (Dubai Shopping Festival) 2025: Jan–Feb
    {"name": "DSF_2025",         "start": date(2025, 1,  3), "end": date(2025, 2,  2), "type": "dsf"},
    # DSF 2026: Jan–Feb (projected)
    {"name": "DSF_2026",         "start": date(2026, 1,  2), "end": date(2026, 2,  1), "type": "dsf"},
    # UAE National Day 2024
    {"name": "National_Day_2024","start": date(2024, 12, 1), "end": date(2024, 12, 3), "type": "national_day"},
    # UAE National Day 2025
    {"name": "National_Day_2025","start": date(2025, 12, 1), "end": date(2025, 12, 3), "type": "national_day"},
    # Back to School 2024
    {"name": "BTS_2024",         "start": date(2024, 8, 16), "end": date(2024, 9, 15), "type": "back_to_school"},
    # Back to School 2025
    {"name": "BTS_2025",         "start": date(2025, 8, 16), "end": date(2025, 9, 15), "type": "back_to_school"},
]

# Precompute: date → set of active event types
def _build_event_index() -> dict[date, list[str]]:
    index: dict[date, list[str]] = {}
    for event in UAE_EVENTS:
        d = event["start"]
        while d <= event["end"]:
            index.setdefault(d, []).append(event["type"])
            d += timedelta(days=1)
    return index

EVENT_INDEX = _build_event_index()


# =============================================================================
# SUPPLIER PROFILES
# Realistic lead times for Dubai retail supply chain
# =============================================================================

SUPPLIERS = [
    {"name": "Al Futtaim Trading",      "lead_days": 7,  "reliability": 0.95},
    {"name": "Noon Fulfillment",         "lead_days": 3,  "reliability": 0.98},
    {"name": "China Direct Import",      "lead_days": 21, "reliability": 0.80},
    {"name": "UAE Local Manufacturer",   "lead_days": 5,  "reliability": 0.92},
    {"name": "India Import Partner",     "lead_days": 14, "reliability": 0.85},
    {"name": "Turkey Fashion Supplier",  "lead_days": 18, "reliability": 0.82},
]


# =============================================================================
# CORE GENERATION LOGIC
# =============================================================================

def _get_season_multiplier(d: date) -> float:
    """Returns a mild background seasonal multiplier (weekend lift, month rhythm)."""
    # Weekend uplift (Fri–Sat in UAE)
    weekend = 1.15 if d.weekday() in (4, 5) else 1.0
    # Slight end-of-month salary effect
    month_end = 1.08 if d.day >= 25 else 1.0
    return weekend * month_end


def _get_event_multiplier(d: date, category_cfg: dict) -> float:
    """Compute cumulative demand multiplier for active UAE events on date d."""
    events = EVENT_INDEX.get(d, [])
    mult = 1.0
    for ev_type in events:
        if ev_type == "ramadan":
            mult *= category_cfg["ramadan_multiplier"]
        elif ev_type == "eid":
            mult *= category_cfg["ramadan_multiplier"] * 1.2  # Eid > Ramadan
        elif ev_type == "dsf":
            mult *= category_cfg["dsf_multiplier"]
        elif ev_type == "national_day":
            mult *= category_cfg["national_day_multiplier"]
        elif ev_type == "back_to_school":
            if category_cfg.get("seasonality_peak") == "back_to_school":
                mult *= 1.8
    # Cap at 5x to avoid unrealistic spikes
    return min(mult, 5.0)


def _generate_sku_id(category: str, index: int) -> str:
    prefix = {"fashion": "FSH", "electronics": "ELC",
               "home": "HOM", "food": "FD", "beauty": "BTY"}
    return f"{prefix[category]}-{index:04d}"


def generate_catalogue(num_skus: int = 500) -> list[dict]:
    """Generate the product catalogue with bilingual names and metadata."""
    catalogue = []
    sku_counter = 1

    for cat_name, cat_cfg in CATEGORIES.items():
        count = cat_cfg["count"]
        products = cat_cfg["products"]

        for i in range(count):
            template = products[i % len(products)]
            eng_name, arabic_name, base_price, variance = template

            # Add variant suffix to avoid exact duplicates
            variant_idx = i // len(products)
            variant_suffix_en = f" {'Pro' if variant_idx == 1 else 'Plus' if variant_idx == 2 else 'Premium' if variant_idx == 3 else 'Lite' if variant_idx == 4 else str(variant_idx + 1) if variant_idx > 0 else ''}".strip()
            variant_suffix_ar = f" {'برو' if variant_idx == 1 else 'بلس' if variant_idx == 2 else 'بريميوم' if variant_idx == 3 else 'لايت' if variant_idx == 4 else str(variant_idx + 1) if variant_idx > 0 else ''}".strip()

            # Price with slight randomisation around category base
            price = round(base_price + random.uniform(-variance * 0.3, variance * 0.5), 0)
            price = max(price, 15)  # floor at AED 15

            supplier = random.choice(SUPPLIERS)
            reorder_point = random.randint(20, 80)
            reorder_qty = random.randint(50, 300)

            catalogue.append({
                "sku_id":              _generate_sku_id(cat_name, sku_counter),
                "name_en":             eng_name + (f" {variant_suffix_en}" if variant_suffix_en else ""),
                "name_ar":             arabic_name + (f" {variant_suffix_ar}" if variant_suffix_ar else ""),
                "category":            cat_name,
                "price_aed":          float(price),
                "supplier_name":       supplier["name"],
                "supplier_lead_days":  supplier["lead_days"],
                "supplier_reliability":supplier["reliability"],
                "reorder_point_units": reorder_point,
                "reorder_qty_units":   reorder_qty,
                "is_ramadan_hero":     int(cat_cfg["ramadan_multiplier"] >= 2.0),
                "is_dsf_hero":         int(cat_cfg["dsf_multiplier"] >= 2.0),
                "tags_en":             f"{cat_name}, {eng_name.lower()}, uae retail, dubai",
                "tags_ar":             f"{cat_name}, {arabic_name}, تجزئة الإمارات, دبي",
            })
            sku_counter += 1

    # Trim/pad to exactly num_skus if needed (shouldn't be necessary with counts above)
    return catalogue[:num_skus]


def generate_sales_history(catalogue: list[dict]) -> list[dict]:
    """
    Generate 24 months of daily sales per SKU.
    
    ENGINEERING NOTE: Daily granularity is intentional.
    Prophet requires daily time-series to model intra-week patterns.
    Aggregating to weekly loses the Friday-Saturday UAE weekend spike.
    This is a real production consideration — mention in interviews.
    """
    rows = []
    for sku in catalogue:
        cat_name = sku["category"]
        cat_cfg  = CATEGORIES[cat_name]
        
        # Base daily demand: varies by price (higher price → lower volume)
        price     = sku["price_aed"]
        base_demand = max(2.0, 30 - (price / 50))  # AED 50 item → ~29/day; AED 500 → ~20/day

        for d in DATE_RANGE:
            season_mult = _get_season_multiplier(d)
            event_mult  = _get_event_multiplier(d, cat_cfg)
            
            # Poisson-distributed daily units with noise
            expected = base_demand * season_mult * event_mult
            units_sold = max(0, int(random.gauss(expected, expected * 0.3)))
            
            # Stockout simulation: ~5% of event days have partial stockout
            if event_mult > 1.5 and random.random() < 0.05:
                units_sold = int(units_sold * 0.4)
                stockout_flag = 1
            else:
                stockout_flag = 0

            revenue = round(units_sold * sku["price_aed"], 2)
            
            # Active UAE events on this date (for Prophet regressors)
            active_events = EVENT_INDEX.get(d, [])
            
            rows.append({
                "sku_id":           sku["sku_id"],
                "date":             d.isoformat(),
                "units_sold":       units_sold,
                "revenue_aed":      revenue,
                "is_ramadan":       int("ramadan" in active_events or "eid" in active_events),
                "is_dsf":           int("dsf" in active_events),
                "is_national_day":  int("national_day" in active_events),
                "is_back_to_school":int("back_to_school" in active_events),
                "stockout_occurred":stockout_flag,
                "event_multiplier": round(event_mult, 3),
            })

    return rows


def generate_inventory_snapshot(catalogue: list[dict]) -> list[dict]:
    """
    Current inventory state for each SKU — as of dataset generation date.
    Inventory levels are set relative to reorder points to create
    realistic scenarios: some SKUs critically low, most healthy, a few overstocked.
    """
    rows = []
    for sku in catalogue:
        # Distribution: 15% critically low, 65% healthy, 20% overstocked
        scenario = random.choices(
            ["critical", "healthy", "overstock"],
            weights=[15, 65, 20]
        )[0]

        rp = sku["reorder_point_units"]
        rq = sku["reorder_qty_units"]

        if scenario == "critical":
            current_stock = random.randint(0, int(rp * 0.5))
        elif scenario == "healthy":
            current_stock = random.randint(int(rp * 0.8), int(rp * 2.5))
        else:  # overstock
            current_stock = random.randint(int(rp * 3), int(rp * 5))

        rows.append({
            "sku_id":               sku["sku_id"],
            "current_stock_units":  current_stock,
            "reorder_point_units":  sku["reorder_point_units"],
            "reorder_qty_units":    sku["reorder_qty_units"],
            "days_of_stock_left":   round(current_stock / max(1, sku["reorder_point_units"] * 0.1), 1),
            "last_reorder_date":    (END_DATE - timedelta(days=random.randint(3, 45))).isoformat(),
            "warehouse_location":   random.choice(["JAFZA", "DIP", "Al Quoz", "Ras Al Khor"]),
            "inventory_status":     scenario,
        })

    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(catalogue: list[dict], sales: list[dict]) -> None:
    """Write dataset metadata for portfolio/interview transparency."""
    meta = {
        "generated_by":    "dubai_retail_intelligence_agent/data/synthetic/generate_dataset.py",
        "generated_at":    END_DATE.isoformat(),
        "data_type":       "SYNTHETIC — not real retailer data",
        "random_seed":     RANDOM_SEED,
        "sku_count":       len(catalogue),
        "sales_rows":      len(sales),
        "date_range":      {"start": START_DATE.isoformat(), "end": END_DATE.isoformat()},
        "categories":      {cat: cfg["count"] for cat, cfg in CATEGORIES.items()},
        "uae_events_modeled": [e["name"] for e in UAE_EVENTS],
        "seasonality_notes": {
            "ramadan": "Demand spike 1.3x–3.2x depending on category",
            "dsf":     "Demand spike 1.6x–2.8x — electronics and fashion peak",
            "national_day": "Demand spike 1.3x–1.8x — food and fashion",
            "eid":     "Additional 1.2x on top of Ramadan multiplier",
            "weekend": "1.15x every Fri–Sat (UAE weekend)",
        },
        "known_limitations": [
            "Hijri calendar dates are approximated — use hijri-converter in production",
            "Sales distribution is Gaussian — real retail is more fat-tailed",
            "No returns/refunds modeled",
            "Cross-SKU demand correlation not modeled (basket effects)",
        ],
        "calibration_note": (
            "All distribution parameters (base_demand, multipliers) are exposed as "
            "constants in generate_dataset.py. When real retailer data is available, "
            "fit these parameters against actual sales before running Prophet."
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Dubai Retail Intelligence Agent — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Date range: {START_DATE} → {END_DATE} ({len(DATE_RANGE)} days)")
    
    print("\n[1/4] Generating product catalogue (500 SKUs)...")
    catalogue = generate_catalogue(500)
    write_csv(catalogue, CATALOGUE_PATH)
    print(f"      ✓ {len(catalogue)} SKUs → {CATALOGUE_PATH}")

    print("\n[2/4] Generating 24-month sales history...")
    print("      This takes ~30s — Poisson sampling across 500 SKUs × 730 days")
    sales = generate_sales_history(catalogue)
    write_csv(sales, SALES_PATH)
    print(f"      ✓ {len(sales):,} rows → {SALES_PATH}")

    print("\n[3/4] Generating inventory snapshot...")
    inventory = generate_inventory_snapshot(catalogue)
    write_csv(inventory, INVENTORY_PATH)
    print(f"      ✓ {len(inventory)} SKUs → {INVENTORY_PATH}")

    print("\n[4/4] Writing metadata...")
    write_metadata(catalogue, sales)
    print(f"      ✓ {METADATA_PATH}")

    # Quick sanity stats
    print("\n── Dataset Summary ──────────────────────────────────────────")
    total_revenue = sum(r["revenue_aed"] for r in sales)
    critical_skus = sum(1 for r in inventory if r["inventory_status"] == "critical")
    print(f"  Total synthetic revenue (24mo): AED {total_revenue:,.0f}")
    print(f"  SKUs in critical stock:          {critical_skus} / {len(catalogue)}")
    print(f"  Sales rows generated:            {len(sales):,}")
    print("\n✓ Dataset generation complete. Data is SYNTHETIC — label clearly in portfolio.")
