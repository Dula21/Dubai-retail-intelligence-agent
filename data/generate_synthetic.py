"""
data/generate_synthetic.py
---------------------------
SYNTHETIC sales history + inventory snapshot for the Dubai Retail Intelligence Agent.
Nothing here is real retailer data. Seeded, so every run is reproducible.

What it models
  demand(day, sku) = baseline(sku) x weekday x payday x trend x UAE event multipliers x noise, drawn Poisson
  Event multipliers come from data/synthetic/uae_retail_calendar.csv + event_lifts.csv, the same
  files the seasonality node and Prophet should read. The catalogue's is_ramadan_hero / is_dsf_hero
  flags add an extra lift on those events.

Inventory is SIMULATED, not invented separately
  Each SKU follows a static reorder policy sized on BASELINE demand (a retailer's gut-feel policy:
  ROP = baseline x (lead_days + 5 safety days), ROQ = baseline x (lead_days + 14/21/30 days)). Seasonal surges
  therefore cause real stockouts, which censor observed sales, as they do in real life. The uncensored
  demand is written to ground_truth_demand.csv for EVALUATING reorder recommendations (never shown to
  the agent): a recommendation can be scored against what customers actually wanted.
  inventory_snapshot.csv is the END STATE of that simulation, so stock, reorder points, days_of_stock_left
  and sales all agree with each other (the earlier files did not: sales implied 5-15x more demand).

Status rule (documented, deterministic):  critical if days_of_stock_left < 7;
  overstock if days_of_stock_left > 35;  otherwise healthy.

Usage (non-destructive, writes to a new folder):
  python data/generate_synthetic.py --out-dir data/synthetic_v2
"""
from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_BASE = {"fashion": 3.0, "electronics": 1.6, "home": 2.4, "beauty": 3.6, "food": 6.0}  # units/day
WEEKDAY = np.array([0.95, 0.92, 0.95, 1.00, 1.12, 1.15, 1.05])        # Mon..Sun (UAE weekend Sat/Sun, Fri half day)
PAYDAY = {"fashion": 1.10, "electronics": 1.10, "beauty": 1.10, "home": 1.05, "food": 1.05}  # month end / start
SALE_DISCOUNT = {"White Friday": 0.20, "Dubai Shopping Festival": 0.10,
                 "New Year / Year-End Sale": 0.15, "Singles Day 11.11": 0.10}
HERO_BOOST, MAX_MULT, WARMUP_DAYS, SAFETY_DAYS = 1.5, 5.0, 60, 5
CRITICAL_DAYS, OVERSTOCK_DAYS = 7, 35
WAREHOUSES = ["DIP", "Ras Al Khor", "JAFZA"]

# Output column names. If your sales loader expects different names, change the right-hand side only.
SALES_COLUMNS = {"date": "date", "sku_id": "sku_id", "units_sold": "units_sold",
                 "unit_price_aed": "unit_price_aed", "revenue_aed": "revenue_aed"}


def _truthy(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"}).to_numpy()


def event_multipliers(cat: pd.DataFrame, lifts: pd.DataFrame, event: str, hero_flag: str) -> np.ndarray:
    """Per-SKU peak multiplier for one event: max over matching lift rows, x HERO_BOOST for hero SKUs."""
    text = (cat["name_en"].fillna("") + " " + cat.get("tags_en", pd.Series("", index=cat.index)).fillna("")).str.lower()
    vals = np.full(len(cat), -np.inf)
    for _, r in lifts[lifts["event"] == event].iterrows():
        cat_ok = (cat["category"] == r["category"]).to_numpy()
        pattern = r["name_regex"] if isinstance(r["name_regex"], str) and r["name_regex"] else None
        kw_ok = text.str.contains(pattern, regex=True).to_numpy() if pattern else np.ones(len(cat), bool)
        vals = np.where(cat_ok & kw_ok, np.maximum(vals, r["multiplier"]), vals)
    m = np.where(np.isinf(vals), 1.0, vals)          # no matching row = no lift
    if hero_flag and hero_flag in cat.columns:
        m = np.where(_truthy(cat[hero_flag]), 1 + (m - 1) * HERO_BOOST + 0.2, m)
    return m


def daily_multiplier(cat, calendar, lifts, dates: pd.DatetimeIndex) -> np.ndarray:
    """(T, N) event multiplier: lead-in ramp, full lift inside the event, post-event dip."""
    mult = np.ones((len(dates), len(cat)))
    cache: dict = {}
    for _, ev in calendar.iterrows():
        start, end = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        lead, dip = int(ev["lead_in_days"]), int(ev["dip_days"])
        if end + pd.Timedelta(days=dip) < dates[0] or start - pd.Timedelta(days=lead) > dates[-1]:
            continue
        key = (ev["event"], ev["hero_flag"] if isinstance(ev["hero_flag"], str) else "")
        if key not in cache:
            cache[key] = event_multipliers(cat, lifts, *key)
        m = cache[key]
        d = (dates - start).days.to_numpy()
        shape = np.zeros(len(dates))
        if lead > 0:
            ramp = (d < 0) & (d >= -lead)
            shape[ramp] = 0.6 * (d[ramp] + lead + 1) / (lead + 1)
        shape[np.asarray((dates >= start) & (dates <= end))] = 1.0
        mult *= 1 + (m[None, :] - 1) * shape[:, None]
        if dip > 0 and ev["dip_factor"] < 1:
            after = np.asarray((dates > end) & (dates <= end + pd.Timedelta(days=dip)))
            depth = (1 - ev["dip_factor"]) * np.clip(m - 1, 0, 1)
            mult[after] *= (1 - depth)[None, :]
    return np.clip(mult, 0.5, MAX_MULT)


def generate(cat: pd.DataFrame, calendar: pd.DataFrame, lifts: pd.DataFrame, end_date: date, days: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(cat)
    total = days + WARMUP_DAYS
    dates_all = pd.date_range(end=pd.Timestamp(end_date), periods=total)

    # baseline demand per SKU: category level, cheaper items sell more, per-SKU spread
    base = cat["category"].map(CATEGORY_BASE).fillna(2.5).to_numpy()
    price = cat["price_aed"].astype(float).to_numpy()
    med = cat.groupby("category")["price_aed"].transform("median").astype(float).to_numpy()
    base = base * np.clip((med / np.maximum(price, 1)) ** 0.4, 0.5, 2.0) * rng.lognormal(0, 0.45, n)

    years = np.arange(total)[:, None] / 365.0
    trend = (1.10 ** years) * np.exp(rng.normal(0, 0.15, n)[None, :] * years)
    dow = WEEKDAY[dates_all.dayofweek.to_numpy()][:, None]
    dom = dates_all.day.to_numpy()
    payday_on = ((dom >= 25) | (dom <= 3))[:, None]
    payday = np.where(payday_on, cat["category"].map(PAYDAY).fillna(1.05).to_numpy()[None, :], 1.0)
    events = daily_multiplier(cat, calendar, lifts, dates_all)
    lam = base[None, :] * dow * payday * trend * events * rng.lognormal(0, 0.12, (total, n))
    demand = rng.poisson(lam)

    # static reorder policy on BASELINE demand (no seasonality: the retailer's gut-feel policy)
    lead = np.clip(cat["supplier_lead_days"].astype(float).to_numpy(), 1, 60).astype(int)
    rel = cat["supplier_reliability"].astype(float).to_numpy()
    rel = np.clip(rel / 100 if rel.max() > 1.5 else rel, 0.5, 1.0)
    rop = np.maximum(5, np.ceil(base * (lead + SAFETY_DAYS))).astype(int)
    cover = lead + rng.choice([14, 21, 30], n)       # an order always covers its own lead time plus 2-4 weeks
    roq = np.maximum(10, 5 * np.ceil(base * cover / 5)).astype(int)

    stock = (rop + rng.random(n) * roq).astype(int)
    arrival = np.full(n, -1)
    on_order = np.zeros(n, int)
    last_order = np.full(n, -1)
    sold_all = np.zeros((total, n), int)
    stock_all = np.zeros((total, n), int)
    for t in range(total):
        arr = arrival == t
        stock[arr] += on_order[arr]
        on_order[arr] = 0
        arrival[arr] = -1
        sold = np.minimum(demand[t], stock)
        stock = stock - sold
        need = (stock <= rop) & (arrival < 0)
        delay = np.where(rng.random(n) > rel, rng.integers(2, 11, n), 0)
        arrival[need] = t + lead[need] + delay[need]
        on_order[need] = roq[need]
        last_order[need] = t
        sold_all[t], stock_all[t] = sold, stock

    sold_w, stock_w = sold_all[WARMUP_DAYS:], stock_all[WARMUP_DAYS:]
    dates = dates_all[WARMUP_DAYS:]

    # discounts on sale events
    disc = np.zeros(days)
    for _, ev in calendar.iterrows():
        if ev["event"] in SALE_DISCOUNT:
            disc[np.asarray((dates >= ev["start_date"]) & (dates <= ev["end_date"]))] = SALE_DISCOUNT[ev["event"]]
    unit_price = np.round(price[None, :] * (1 - disc[:, None]), 2)

    # events active on each date (date-level file, 1 row per day): lets a node compute a SKU's peak
    # period from data instead of a hard-coded list. Lead-in and dip days are not "active".
    # Overlapping events stack (e.g. Mother's Day inside Eid al-Fitr in 2026), so peak attribution is approximate.
    active = np.array(["|".join(ev["event"] for _, ev in calendar.iterrows()
                                if pd.Timestamp(ev["start_date"]) <= d <= pd.Timestamp(ev["end_date"]))
                       for d in dates])
    S = SALES_COLUMNS
    sales = pd.DataFrame({
        S["date"]: np.repeat(dates.strftime("%Y-%m-%d"), n),
        S["sku_id"]: np.tile(cat["sku_id"].to_numpy(), days),
        S["units_sold"]: sold_w.ravel(),
        S["unit_price_aed"]: unit_price.ravel(),
        S["revenue_aed"]: np.round(sold_w * unit_price, 2).ravel(),
    })
    # EVALUATION ONLY: true demand before stockouts censored it. Never feed this to the agent.
    truth = pd.DataFrame({"date": sales[S["date"]], "sku_id": sales[S["sku_id"]],
                          "demand_units": demand[WARMUP_DAYS:].ravel()})

    cur = stock_w[-1]
    avg30 = sold_w[-30:].mean(axis=0)
    days_left = np.round(np.clip(cur / np.maximum(avg30, 0.05), 0, 120), 1)
    status = np.where(days_left < CRITICAL_DAYS, "critical", np.where(days_left > OVERSTOCK_DAYS, "overstock", "healthy"))
    last_idx = np.where(last_order >= WARMUP_DAYS, last_order - WARMUP_DAYS, 0)
    inventory = pd.DataFrame({
        "sku_id": cat["sku_id"].to_numpy(),
        "current_stock_units": cur,
        "reorder_point_units": rop,
        "reorder_qty_units": roq,
        "days_of_stock_left": days_left,
        "last_reorder_date": dates[last_idx].strftime("%Y-%m-%d"),
        "warehouse_location": rng.choice(WAREHOUSES, n, p=[0.35, 0.30, 0.35]),
        "inventory_status": status,
    })
    date_events = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "active_events": active})
    return sales, inventory, truth, date_events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=str(ROOT / "data/synthetic/product_catalogue.csv"))
    ap.add_argument("--calendar", default=str(ROOT / "data/synthetic/uae_retail_calendar.csv"))
    ap.add_argument("--lifts", default=str(ROOT / "data/synthetic/event_lifts.csv"))
    ap.add_argument("--end-date", default=date.today().isoformat())
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "data/synthetic_v2"))
    a = ap.parse_args()

    cat = pd.read_csv(a.catalogue)
    sales, inv, truth, date_events = generate(cat, pd.read_csv(a.calendar), pd.read_csv(a.lifts), date.fromisoformat(a.end_date), a.days, a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sales.to_csv(out / "sales_history.csv", index=False)
    inv.to_csv(out / "inventory_snapshot.csv", index=False)
    date_events.to_csv(out / "date_events.csv", index=False)
    truth.to_csv(out / "ground_truth_demand.csv", index=False)   # eval only; consider .gitignore (about 10 MB)
    print(f"sales_history.csv: {len(sales):,} rows | inventory_snapshot.csv: {len(inv)} rows -> {out}")
    print(inv["inventory_status"].value_counts().to_string())
    print("days_of_stock_left:", inv["days_of_stock_left"].describe().round(1).to_dict())


if __name__ == "__main__":
    main()