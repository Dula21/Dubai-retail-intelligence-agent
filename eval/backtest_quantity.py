"""
eval/backtest_quantity.py
--------------------------
Time-split backtest of the order-quantity engine against TRUE demand.

EVALUATION ONLY: this script reads data/synthetic/ground_truth_demand.csv (uncensored demand). The agent never
does (tests/test_quantity_engine.py fails if anything under backend/ references it).

What is measured, and why it is defined this way (fixed BEFORE running, not tuned after)
  One CASE = (SKU, past calendar-event occurrence).
  Order date T = event start - lead-in days - supplier lead days - ORDER_BUFFER_DAYS  (the seasonality node's order-by rule).
  Cover window = cover_window(T, lead, event): from max(arrival, lead-in start) to event end.
  Every forecaster is fitted ONLY on sales up to T. Truth = sum of true demand over the cover window.
  A case is eligible only if the SAME event occurred earlier, fully inside the training data, so the effect
  is learnable from history; events seen once (e.g. Navratri) are skipped, not scored.

  Target stock level = window forecast x conformal safety multiplier (calibrated per forecaster).
  Calibration cases (window ended before the split date) and test cases (order date on/after the split date) never overlap.

  Metrics on test cases
    coverage      share of cases with truth <= target (no stockout over the window)      target: the service level
    precision     share with truth <= target <= (1 + TOLERANCE) x truth                   (no stockout AND not wasteful)
                  headline tolerance 0.30; 0.20 and 0.50 are shown so the trade-off is visible
    WAPE / bias   accuracy of the raw window forecast (before the safety multiplier)

Forecasters compared
  prophet        engine default: censoring-aware (mask mode below) + one learned coefficient per event window
  lift_table     baseline x the calendar LIFT TABLE (what Phase 2 knew). The data was generated from that table,
                 so it is a circular, favourable comparison: it must not be read as real-world accuracy
  naive          flat mean of the last 28 uncensored days

CAVEAT to repeat wherever a number from this script is quoted: the data is synthetic and its lifts come from
event_lifts.csv. This validates the PIPELINE; it says nothing about real Dubai demand.

Usage (from the repo root):
  python eval/backtest_quantity.py --sample-skus 100 --workers 2
  python eval/backtest_quantity.py --sample-skus 100 --workers 2 --compare-masking
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

TOLERANCES = (0.20, 0.30, 0.50)
HEADLINE_TOLERANCE = 0.30
LIFT_TABLE_LEAD_IN_SHARE = 0.5       # lift_table forecaster: lead-in days get half of the event's lift (a plain assumption)
BASELINE_DAYS = 56

# ── worker state (loaded once per process) ─────────────────────────────────────
_W: dict = {}


def _init(data_dir: str, mask_modes: tuple, min_train_days: int) -> None:
    os.environ["CALENDAR_DATA_PATH"] = str(Path(data_dir) / "uae_retail_calendar.csv")
    os.environ["EVENT_LIFTS_DATA_PATH"] = str(Path(data_dir) / "event_lifts.csv")
    from agents.nodes import seasonality_node as sn
    from services import quantity_engine as qe
    d = Path(data_dir)
    sales = pd.read_csv(d / "sales_history.csv", parse_dates=["date"])
    truth = pd.read_csv(d / "ground_truth_demand.csv", parse_dates=["date"])
    cat = pd.read_csv(d / "product_catalogue.csv").set_index("sku_id")
    cal = pd.read_csv(d / "uae_retail_calendar.csv", parse_dates=["start_date", "end_date"])
    events, _ = sn._load_events()
    _W.update(sn=sn, qe=qe, cal=cal, cat=cat, events=events, mask_modes=mask_modes, min_train_days=min_train_days,
              sales={k: g.sort_values("date").reset_index(drop=True) for k, g in sales.groupby("sku_id")},
              truth={k: g.sort_values("date").set_index("date")["demand_units"] for k, g in truth.groupby("sku_id")},
              history_start=sales["date"].min(), history_end=sales["date"].max())


def _lift_table_forecast(series: pd.DataFrame, category: str, text: str, T: pd.Timestamp):
    sn, qe, cal, events = _W["sn"], _W["qe"], _W["cal"], _W["events"]
    regs = qe.event_regressors(series["ds"], cal)
    in_event = regs[[c for c in regs.columns if not c.endswith("_dip")]].max(axis=1).to_numpy() > 0 if len(regs.columns) else np.zeros(len(series), bool)
    tail = series.tail(BASELINE_DAYS)
    ok = tail["y"].notna().to_numpy() & ~in_event[-len(tail):]
    pool = tail["y"].to_numpy()[ok] if ok.sum() >= 7 else tail["y"].dropna().to_numpy()
    if len(pool) < 7:
        return None
    base = float(pool.mean())

    def predict(idx: pd.DatetimeIndex) -> np.ndarray:
        mult = np.ones(len(idx))
        for e in events:
            m = sn.lift_for(e, category, text)
            if m <= 1.0:
                continue
            lead = pd.Timestamp(e.start) - pd.Timedelta(days=e.lead_in_days)
            inside = np.asarray((idx >= pd.Timestamp(e.start)) & (idx <= pd.Timestamp(e.end)))
            pre = np.asarray((idx >= lead) & (idx < pd.Timestamp(e.start)))
            mult = mult * np.where(inside, m, 1.0) * np.where(pre, 1 + (m - 1) * LIFT_TABLE_LEAD_IN_SHARE, 1.0)
        return base * mult

    return qe.Forecast(method="lift_table", train_end=T, n_train_days=int(series["y"].notna().sum()),
                       masked_share=0.0, _predict=predict)


def _cases_for_sku(sku: str) -> list[dict]:
    sn, qe, cal, cat = _W["sn"], _W["qe"], _W["cal"], _W["cat"]
    row = cat.loc[sku]
    lead = int(row["supplier_lead_days"])
    category = str(row["category"])
    text = f"{row['name_en']} {row.get('tags_en', '')}"
    sales_sku, truth_sku = _W["sales"][sku], _W["truth"][sku]
    h0, h1 = _W["history_start"], _W["history_end"]
    ev_objs = {(e.name, pd.Timestamp(e.start)): e for e in _W["events"]}
    out: list[dict] = []

    for r in cal.sort_values("start_date").itertuples(index=False):
        start, end, lead_in = pd.Timestamp(r.start_date), pd.Timestamp(r.end_date), int(r.lead_in_days)
        if end > h1:
            continue
        T = start - pd.Timedelta(days=lead_in + lead + qe.ORDER_BUFFER_DAYS)
        if (T - h0).days < _W["min_train_days"]:
            continue
        prior = cal[(cal["event"] == r.event) & (cal["end_date"] <= T) & (cal["start_date"] - pd.to_timedelta(cal["lead_in_days"], unit="D") >= h0)]
        if prior.empty:
            continue                                             # nothing to learn this event from
        win = qe.cover_window(T.date(), lead, {"start": start.date(), "end": end.date(), "lead_in_days": lead_in})
        if win is None:
            continue
        D = float(truth_sku.loc[win.start:win.end].sum())
        if D <= 0:
            continue
        ev = ev_objs.get((r.event, start))
        lifted = bool(ev and sn.lift_for(ev, category, text) >= sn.MIN_LIFT)

        rec = dict(sku=sku, category=category, event=r.event, event_start=str(start.date()), T=str(T.date()),
                   window_start=str(win.start.date()), window_end=str(win.end.date()), lead=lead, truth=D, lifted=lifted)
        for mode in _W["mask_modes"]:
            ser = qe.demand_series(sales_sku, mode, cutoff=T)
            fc = qe.fit_prophet(ser, cal, mode)
            if fc is not None:
                rec[f"F_prophet_{mode}"] = fc.window_sum(win.start, win.end)
                if mode == _W["mask_modes"][0]:
                    rec["masked_share"] = fc.masked_share
                    rec["events_without_history"] = bool(set(fc.events_without_history) & set(qe.event_names_in_window(cal, win.start, win.end)))
        ser_run = qe.demand_series(sales_sku, "run", cutoff=T)
        nv = qe.fit_naive(ser_run)
        if nv is not None:
            rec["F_naive"] = nv.window_sum(win.start, win.end)
        lt = _lift_table_forecast(ser_run, category, text, T)
        if lt is not None:
            rec["F_lift_table"] = lt.window_sum(win.start, win.end)
        out.append(rec)
    return out


# ── scoring ────────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, fcol: str, level: float, split: pd.Timestamp) -> dict:
    from services import quantity_engine as qe
    d = df.dropna(subset=[fcol]).copy()
    cal = d[pd.to_datetime(d["window_end"]) <= split]
    test = d[pd.to_datetime(d["T"]) >= split]
    if len(cal) < 30 or len(test) < 30:
        return {"forecaster": fcol, "error": f"too few cases (calibration {len(cal)}, test {len(test)})"}
    sm = qe.SafetyMultiplier.fit(cal[fcol].to_numpy(), cal["truth"].to_numpy(), level)
    test = test.assign(mult=[sm.multiplier(f) for f in test[fcol]])
    test["target"] = test[fcol] * test["mult"]

    def block(t: pd.DataFrame) -> dict:
        n = len(t)
        if n == 0:
            return {"n": 0}
        cov = float((t["truth"] <= t["target"]).mean())
        out = {"n": n, "coverage": round(cov, 3),
               "wape": round(float((t[fcol] - t["truth"]).abs().sum() / t["truth"].sum()), 3),
               "bias": round(float((t[fcol].sum() - t["truth"].sum()) / t["truth"].sum()), 3),
               "median_overshoot_when_covered": round(float(((t["target"] / t["truth"] - 1)[t["truth"] <= t["target"]]).median()), 3)}
        for tol in TOLERANCES:
            out[f"precision@{int(tol * 100)}"] = round(float(((t["truth"] <= t["target"]) & (t["target"] <= (1 + tol) * t["truth"])).mean()), 3)
        return out

    res = {"forecaster": fcol, "calibration_cases": len(cal), "test_cases": len(test), "safety": sm.to_dict(),
           "test_all": block(test), "test_lifted_skus": block(test[test["lifted"]]),
           "by_category_lifted": {c: block(g) for c, g in test[test["lifted"]].groupby("category")}}
    if "events_without_history" in test:
        res["test_lifted_events_without_history"] = block(test[test["lifted"] & test["events_without_history"].fillna(False).astype(bool)])
    return res


def masking_report(df: pd.DataFrame, modes: tuple) -> dict:
    out = {}
    for mode in modes:
        col = f"F_prophet_{mode}"
        d = df.dropna(subset=[col])
        out[mode] = {"n": len(d),
                     "wape": round(float((d[col] - d["truth"]).abs().sum() / d["truth"].sum()), 3),
                     "bias": round(float((d[col].sum() - d["truth"].sum()) / d["truth"].sum()), 3),
                     "bias_lifted_skus": round(float((d[d.lifted][col].sum() - d[d.lifted]["truth"].sum()) / d[d.lifted]["truth"].sum()), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data/synthetic"))
    ap.add_argument("--sample-skus", type=int, default=100, help="stratified by category; 0 = all SKUs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--service-level", type=float, default=0.90)
    ap.add_argument("--min-train-days", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--compare-masking", action="store_true", help="also fit Prophet with mask modes none / flag / run")
    ap.add_argument("--out", default=str(ROOT / "eval/quantity_backtest.json"))
    a = ap.parse_args()

    cat = pd.read_csv(Path(a.data_dir) / "product_catalogue.csv")
    skus = cat["sku_id"].tolist()
    if a.sample_skus and a.sample_skus < len(skus):
        frac = a.sample_skus / len(skus)
        skus = cat.groupby("category", group_keys=False).apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=a.seed))["sku_id"].tolist()
    modes = ("run", "none", "flag") if a.compare_masking else ("run",)
    print(f"{len(skus)} SKUs, mask modes {modes}, {a.workers} workers")

    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.data_dir, modes, a.min_train_days)) as ex:
        rows = [r for part in ex.map(_cases_for_sku, skus, chunksize=4) for r in part]
    df = pd.DataFrame(rows)
    print(f"{len(df)} cases in {time.time() - t0:.0f}s")
    if df.empty:
        sys.exit("no eligible cases")

    split = pd.to_datetime(df["T"]).median()
    print(f"split date {split.date()} (calibrate on windows ended before it; test on order dates on/after it)")
    result = {"data_dir": a.data_dir, "skus": len(skus), "cases": len(df), "split_date": str(split.date()),
              "service_level": a.service_level, "headline_tolerance": HEADLINE_TOLERANCE,
              "caveat": "synthetic data; lifts generated from event_lifts.csv; validates the pipeline, not real Dubai demand",
              "forecasters": {}}
    for fcol in ("F_prophet_run", "F_lift_table", "F_naive"):
        if fcol in df:
            result["forecasters"][fcol] = score(df, fcol, a.service_level, split)
    if a.compare_masking:
        result["masking"] = masking_report(df, modes)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    df.to_csv(Path(a.out).with_suffix(".cases.csv"), index=False)

    print("\n| forecaster | scope | n | coverage | precision@30 | precision@20 | precision@50 | WAPE | bias |\n|---|---|---|---|---|---|---|---|---|")
    for name, r in result["forecasters"].items():
        if "error" in r:
            print(f"| {name} | {r['error']} |"); continue
        for scope in ("test_lifted_skus", "test_all"):
            b = r[scope]
            if b["n"]:
                print(f"| {name} | {scope} | {b['n']} | {b['coverage']:.0%} | {b['precision@30']:.0%} | {b['precision@20']:.0%} | "
                      f"{b['precision@50']:.0%} | {b['wape']:.0%} | {b['bias']:+.0%} |")
    if a.compare_masking:
        print("\nmasking (Prophet raw-forecast accuracy, all cases):", json.dumps(result["masking"]))
    print(f"\nservice-level target {a.service_level:.0%}; wrote {a.out}")


if __name__ == "__main__":
    main()
