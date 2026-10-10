"""
tests/test_synthetic_data.py
-----------------------------
The synthetic data must be internally consistent, otherwise the agent's reorder logic is judged on
nonsense (earlier files: sales implied 5-15x more demand than the inventory snapshot). These tests use a
small generated catalogue plus the real calendar/lift CSVs, so they run in seconds and need no model.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"

spec = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "data" / "generate_synthetic.py")
gen = importlib.util.module_from_spec(spec)
sys.modules["generate_synthetic"] = gen
spec.loader.exec_module(gen)

CAL_PATH, LIFT_PATH = SYN / "uae_retail_calendar.csv", SYN / "event_lifts.csv"
pytestmark = pytest.mark.skipif(not (CAL_PATH.exists() and LIFT_PATH.exists()), reason="calendar CSVs missing")

END = pd.Timestamp("2026-10-08").date()


def make_catalogue():
    names = {"fashion": ["Summer Dress", "Abaya Black", "Winter Coat", "Backpack"],
             "electronics": ["Bluetooth Speaker", "Smart Watch", "Power Bank"],
             "home": ["Ramadan Lantern", "Scented Candles Set", "Camping Tent"],
             "beauty": ["Oud Perfume", "Sunscreen SPF50", "Gift Set"],
             "food": ["Medjool Dates 1kg", "Saffron", "Mixed Nuts"]}
    pre = {"fashion": "FSH", "electronics": "ELC", "home": "HOM", "beauty": "BTY", "food": "FD"}
    rows, i = [], 1
    for cat, ns in names.items():
        for n in ns:
            for v in ["", " Pro", " Lite", " Plus", " 2", " Premium", " Mini", " Max"]:
                rows.append(dict(sku_id=f"{pre[cat]}-{i:04d}", name_en=n + v, category=cat, price_aed=50.0 + i,
                                 supplier_lead_days=7 + (i * 3) % 30, supplier_reliability=0.9,
                                 is_ramadan_hero=("Ramadan" in n or "Dates" in n), is_dsf_hero=("Smart" in n),
                                 tags_en=""))
                i += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    cat = make_catalogue()
    cal, lifts = pd.read_csv(CAL_PATH), pd.read_csv(LIFT_PATH)
    sales, inv, truth, date_events = gen.generate(cat, cal, lifts, END, 730, seed=7)
    return cat, cal, lifts, sales, inv, truth, date_events


# ── calendar ──────────────────────────────────────────────────────────────────

def test_calendar_is_well_formed():
    cal, lifts = pd.read_csv(CAL_PATH), pd.read_csv(LIFT_PATH)
    assert (pd.to_datetime(cal["end_date"]) >= pd.to_datetime(cal["start_date"])).all()
    assert set(cal["event"]) == set(lifts["event"])          # every event has lifts, every lift an event
    assert cal["date_basis"].isin({"computed_umm_al_qura", "computed_holidays_lib", "fixed", "rule",
                                   "estimate_verify"}).all()
    assert {"Ramadan", "Eid al-Fitr", "Diwali", "Christmas", "White Friday", "UAE National Day",
            "Dubai Shopping Festival", "Back to School"} <= set(cal["event"])


def test_diwali_2026_and_ramadan_2027_dates():
    cal = pd.read_csv(CAL_PATH)
    diwali = cal[(cal.event == "Diwali") & cal.start_date.str.startswith("2026")].iloc[0]
    assert diwali.start_date <= "2026-11-08" <= diwali.end_date
    ramadan = cal[(cal.event == "Ramadan") & cal.start_date.str.startswith("2027")].iloc[0]
    assert ramadan.start_date == "2027-02-08"


# ── generator ─────────────────────────────────────────────────────────────────

def test_schema_and_shapes(data):
    cat, _, _, sales, inv, truth, _ = data
    assert len(sales) == 730 * len(cat) and len(inv) == len(cat) and len(truth) == len(sales)
    assert list(inv.columns) == ["sku_id", "current_stock_units", "reorder_point_units", "reorder_qty_units",
                                 "days_of_stock_left", "last_reorder_date", "warehouse_location", "inventory_status"]
    assert sales["date"].max() == END.isoformat()
    assert (sales["units_sold"] >= 0).all() and (inv["current_stock_units"] >= 0).all()


def test_same_seed_same_data(data):
    cat, cal, lifts, sales, inv, *_ = data
    sales2, inv2, *_ = gen.generate(cat, cal, lifts, END, 730, seed=7)
    assert sales.equals(sales2) and inv.equals(inv2)


def test_sales_never_exceed_true_demand(data):
    cat, cal, lifts, sales, inv, truth, _ = data
    assert (sales["units_sold"].to_numpy() <= truth["demand_units"].to_numpy()).all()


def test_hero_and_keyword_skus_lift_more_than_plain_category(data):
    cat, cal, lifts, *_ = data
    dates = pd.date_range("2026-02-01", "2026-04-01")
    mult = gen.daily_multiplier(cat, cal, lifts, dates)
    inside = dates.get_loc(pd.Timestamp("2026-03-05"))          # mid Ramadan 2026 (18 Feb - 19 Mar)
    lantern = cat.index[cat.name_en == "Ramadan Lantern"][0]
    sunscreen = cat.index[cat.name_en == "Sunscreen SPF50"][0]
    dates_out = dates.get_loc(pd.Timestamp("2026-02-01"))
    assert mult[inside, lantern] > 2.5                           # keyword + hero
    assert mult[inside, lantern] > mult[inside, sunscreen]
    assert mult[dates_out, lantern] == pytest.approx(1.0, abs=0.35)


def test_every_catalogue_category_has_multiple_events_with_lift(data):
    cat, cal, lifts, *_ = data
    dates = pd.date_range("2025-01-01", "2026-12-31")
    mult = gen.daily_multiplier(cat, cal, lifts, dates)
    for c in cat["category"].unique():
        cols = (cat["category"] == c).to_numpy()
        assert (mult[:, cols].max(axis=0) > 1.4).all(), c       # nothing is flat all year


def test_inventory_is_consistent_with_sales(data):
    cat, _, _, sales, inv, *_ = data
    last30 = sales[sales["date"] > (pd.Timestamp(END) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")]
    avg30 = last30.groupby("sku_id")["units_sold"].mean().reindex(inv["sku_id"]).to_numpy()
    expected = np.round(np.clip(inv["current_stock_units"].to_numpy() / np.maximum(avg30, 0.05), 0, 120), 1)
    assert np.allclose(inv["days_of_stock_left"].to_numpy(), expected, atol=0.11)


def test_status_rule_and_plausible_mix(data):
    inv = data[4]
    d = inv["days_of_stock_left"]
    assert (inv.loc[d < 7, "inventory_status"] == "critical").all()
    assert (inv.loc[d > 35, "inventory_status"] == "overstock").all()
    mix = inv["inventory_status"].value_counts(normalize=True)
    assert 0.03 < mix.get("critical", 0) < 0.45 and mix.get("healthy", 0) > 0.3     # not "everything critical"


def test_date_events_names_real_events(data):
    cal, date_events = data[1], data[6]
    assert len(date_events) == 730
    seen = set("|".join(date_events["active_events"].fillna("").unique()).split("|")) - {""}
    assert seen and seen <= set(cal["event"])
    day = date_events.loc[date_events["date"] == "2025-11-28", "active_events"].iloc[0]       # White Friday 2025
    assert "White Friday" in day