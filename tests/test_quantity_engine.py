"""
tests/test_quantity_engine.py
------------------------------
The deterministic order-quantity engine: censoring masks, calendar regressors, cover window, conformal
safety multiplier, and the netting arithmetic. No model, no network. One Prophet smoke test is skipped
when prophet is not installed.
"""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services import quantity_engine as qe  # noqa: E402


def days(n, start="2026-01-01"):
    return pd.date_range(start, periods=n)


# ── censoring ──────────────────────────────────────────────────────────────────

def frame(units, flags):
    return pd.Series(units), pd.Series(flags)


def test_none_mode_masks_nothing():
    u, f = frame([5, 0, 0, 4], [0, 1, 1, 0])
    assert not qe.censored_days(u, f, "none").any()


def test_flag_mode_masks_only_flagged_days():
    u, f = frame([5, 3, 0, 0, 4], [0, 1, 0, 1, 0])
    assert qe.censored_days(u, f, "flag").tolist() == [False, True, False, True, False]


def test_run_mode_masks_the_zero_days_inside_a_stockout_run():
    # day 2 sold 0 and is NOT flagged (demand was 0), but it sits inside a stockout run: keeping it would
    # keep only the zero-demand days of a run whose other days are censored.
    u, f = frame([5, 3, 0, 0, 4, 0, 6], [0, 1, 0, 1, 0, 0, 0])
    got = qe.censored_days(u, f, "run").tolist()
    assert got == [False, True, True, True, False, False, False]    # lone zero on day 5 touches no flag: kept


def test_run_mode_keeps_zero_sales_days_far_from_any_stockout():
    u, f = frame([2, 0, 3, 0, 0, 1], [0, 0, 0, 0, 0, 0])
    assert not qe.censored_days(u, f, "run").any()


def test_unknown_mask_mode_is_rejected():
    with pytest.raises(ValueError):
        qe.censored_days(*frame([1], [0]), "bogus")


def test_demand_series_sets_masked_days_to_nan_and_respects_cutoff():
    df = pd.DataFrame({"date": days(6), "units_sold": [3, 3, 1, 3, 3, 3], "stockout_occurred": [0, 0, 1, 0, 0, 0]})
    out = qe.demand_series(df, "flag", cutoff=pd.Timestamp("2026-01-05"))
    assert len(out) == 5 and out["y"].isna().tolist() == [False, False, True, False, False]


# ── calendar regressors ────────────────────────────────────────────────────────

CAL = pd.DataFrame([
    dict(event="Ramadan", start_date="2026-02-18", end_date="2026-03-19", lead_in_days=14, dip_days=0, dip_factor=1.0),
    dict(event="Eid al-Fitr", start_date="2026-03-20", end_date="2026-03-22", lead_in_days=7, dip_days=5, dip_factor=0.85),
    dict(event="Ramadan", start_date="2027-02-08", end_date="2027-03-09", lead_in_days=14, dip_days=0, dip_factor=1.0),
])


def test_regressor_windows_lead_in_event_and_dip():
    idx = pd.DatetimeIndex(["2026-02-03", "2026-02-04", "2026-02-18", "2026-03-19", "2026-03-23", "2026-03-27", "2026-03-28"])
    r = qe.event_regressors(idx, CAL)
    assert r["ev_ramadan_lead"].tolist() == [0, 1, 0, 0, 0, 0, 0]          # 14 days before the 18th starts on the 4th
    assert r["ev_ramadan_in"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert r["ev_eid_al_fitr_dip"].tolist() == [0, 0, 0, 0, 1, 1, 0]       # 23..27 March = 5 days after the 22nd
    assert "ev_ramadan_dip" not in r.columns                               # dip_factor 1.0 -> no dip column


def test_all_occurrences_of_an_event_share_one_column():
    idx = pd.DatetimeIndex(["2026-02-20", "2027-02-10", "2026-12-01"])
    assert qe.event_regressors(idx, CAL)["ev_ramadan_in"].tolist() == [1, 1, 0]


def test_event_names_in_window_includes_lead_in():
    assert qe.event_names_in_window(CAL, pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-12")) == ["Ramadan"]
    assert qe.event_names_in_window(CAL, pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30")) == []


# ── cover window ───────────────────────────────────────────────────────────────

EV = {"start": date(2026, 3, 20), "end": date(2026, 3, 22), "lead_in_days": 7}


def test_event_window_starts_at_lead_in_when_order_is_early():
    w = qe.cover_window(date(2026, 1, 1), 14, EV)
    assert (w.start.date(), w.end.date()) == (date(2026, 3, 13), date(2026, 3, 22))


def test_event_window_starts_at_arrival_when_order_is_late():
    w = qe.cover_window(date(2026, 3, 10), 5, EV)             # arrives 15 March: after the lead-in began (13th)
    assert (w.start.date(), w.end.date()) == (date(2026, 3, 15), date(2026, 3, 22))


def test_order_that_would_arrive_after_the_event_has_no_window():
    assert qe.cover_window(date(2026, 3, 10), 14, EV) is None   # arrives 24 March, event ended on the 22nd


def test_finished_event_has_no_window():
    assert qe.cover_window(date(2026, 4, 1), 7, EV) is None


def test_non_event_window_is_arrival_plus_review_period():
    w = qe.cover_window(date(2026, 5, 1), 10)
    assert (w.start.date(), w.end.date()) == (date(2026, 5, 11), date(2026, 5, 11 + qe.REVIEW_PERIOD_DAYS - 1))


# ── conformal safety multiplier ────────────────────────────────────────────────

def test_conformal_quantile_is_finite_sample_corrected():
    e = np.arange(1, 10)                                       # 9 errors: 1..9
    assert qe.conformal_quantile(e, 0.9) == 9                  # ceil(10*0.9) = 9th smallest
    assert qe.conformal_quantile(e, 0.5) == 5                  # ceil(10*0.5) = 5th smallest
    assert qe.conformal_quantile(e, 0.999) == 9                # k capped at n


def test_relative_error_floors_tiny_forecasts():
    assert qe.relative_errors([0.0], [3.0])[0] == pytest.approx(3.0 / qe.FORECAST_FLOOR_UNITS)


def test_multiplier_gives_at_least_the_service_level_on_exchangeable_data():
    rng = np.random.default_rng(0)
    f = rng.uniform(20, 200, 4000)
    a = rng.poisson(f)
    cal, test = slice(0, 2000), slice(2000, 4000)
    sm = qe.SafetyMultiplier.fit(f[cal], a[cal], 0.9)
    covered = np.mean([a[test][i] <= f[test][i] * sm.multiplier(f[test][i]) for i in range(2000)])
    assert covered >= 0.88                                      # 0.90 target, sampling noise allowed


def test_small_bins_fall_back_to_the_global_quantile():
    f = np.concatenate([np.full(100, 10.0), np.full(5, 500.0)])
    a = np.concatenate([np.full(100, 12.0), np.full(5, 900.0)])
    sm = qe.SafetyMultiplier.fit(f, a, 0.9, n_bins=2)
    assert min(sm.n) < qe.MIN_PER_BIN
    assert sm.q[int(np.argmin(sm.n))] == sm.global_q


def test_safety_multiplier_round_trips_through_json_dict():
    sm = qe.SafetyMultiplier.fit(np.linspace(10, 100, 300), np.linspace(10, 100, 300) * 1.1, 0.9)
    assert qe.SafetyMultiplier.from_dict(sm.to_dict()) == sm


# ── the quantity arithmetic ────────────────────────────────────────────────────

def flat_forecast(per_day, train_end="2026-01-31", masked=0.0, unseen=()):
    return qe.Forecast(method="flat", train_end=pd.Timestamp(train_end), n_train_days=400, masked_share=masked,
                       events_without_history=list(unseen), _predict=lambda idx: np.full(len(idx), float(per_day)))


SAFETY = qe.SafetyMultiplier(0.9, (), (0.20,), (500,), 0.20, 500)
WIN = qe.CoverWindow(pd.Timestamp("2026-02-11"), pd.Timestamp("2026-02-20"), "test")          # 10 days


def run(**kw):
    args = dict(sku_id="X-1", forecast=flat_forecast(10), window=WIN, safety=SAFETY, on_hand=30,
                snapshot_date=date(2026, 1, 31), as_of=date(2026, 1, 31), open_orders=0)
    args.update(kw)
    return qe.order_quantity(**args)


def test_quantity_is_forecast_times_safety_minus_projected_stock_minus_open_orders():
    # forecast 10 days x 10 = 100; x1.2 = 120. snapshot 31 Jan -> window starts 11 Feb: 10 days (1..10 Feb) deplete 100 > 30 stock
    r = run()
    assert r.forecast_units == 100 and r.target_units == 120
    assert r.projected_on_hand == 0 and r.order_qty == 120
    assert "stock_projected_without_replenishment" in r.flags


def test_stock_covers_part_of_the_need_when_window_starts_right_away():
    w = qe.CoverWindow(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"), "test")
    r = run(window=w, on_hand=50)
    assert r.projected_on_hand == 50 and r.order_qty == 70                    # 120 - 50
    assert "stock_projected_without_replenishment" not in r.flags


def test_open_orders_are_netted_and_known_zero_is_not_flagged():
    w = qe.CoverWindow(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"), "test")
    assert run(window=w, on_hand=50, open_orders=40).order_qty == 30
    assert "open_orders_unknown" not in run(window=w, on_hand=50, open_orders=0).flags


def test_missing_open_orders_are_assumed_zero_and_flagged():
    r = run(open_orders=None)
    assert r.open_orders == 0 and "open_orders_unknown" in r.flags


def test_enough_stock_means_order_zero_not_negative():
    w = qe.CoverWindow(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"), "test")
    assert run(window=w, on_hand=500).order_qty == 0


def test_quantity_rounds_up():
    assert run(forecast=flat_forecast(10.04), window=qe.CoverWindow(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"), "t"),
               on_hand=0).order_qty == math.ceil(10.04 * 10 * 1.2)


def test_stale_snapshot_is_flagged():
    r = run(as_of=date(2026, 2, 9))
    assert r.snapshot_age_days == 9 and "stock_snapshot_stale" in r.flags
    assert "stock_snapshot_stale" not in run().flags


def test_unknown_stock_gives_no_quantity_not_zero():
    for bad in (None, -1):
        r = run(on_hand=bad)
        assert r.order_qty is None and r.flags == ["stock_unknown"]


def test_no_forecast_or_calibration_or_window_gives_no_quantity():
    assert run(forecast=None).flags == ["insufficient_history"]
    assert run(safety=None).flags == ["no_calibration"]
    assert run(window=None).flags == ["no_cover_window"]
    assert all(run(**{k: None}).order_qty is None for k in ("forecast", "safety", "window"))


def test_events_without_history_in_the_window_are_flagged():
    r = run(forecast=flat_forecast(10, unseen=["Ramadan"]), calendar=CAL,
            window=qe.CoverWindow(pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19"), "t"))
    assert any(f.startswith("events_without_history:Ramadan") for f in r.flags)


def test_high_masked_share_is_flagged():
    assert "high_masked_share" in run(forecast=flat_forecast(10, masked=0.4)).flags


def test_trace_states_every_input():
    line = run().trace[0]
    for part in ("forecast 100.0", "safety 1.2", "service level 90%", "stock 30", "projected", "open orders 0", "order 120"):
        assert part in line


# ── guard: the evaluation-only file must never reach the agent ─────────────────

def test_backend_never_references_ground_truth():
    offenders = [str(p.relative_to(ROOT)) for p in (ROOT / "backend").rglob("*.py")
                 if "ground_truth" in p.read_text(encoding="utf-8")
                 and p.name != "quantity_engine.py"]                       # this module only names it in its docstring
    assert not offenders, f"backend modules reference ground_truth_demand: {offenders}"


def test_engine_only_mentions_ground_truth_in_its_docstring():
    src = (ROOT / "backend/services/quantity_engine.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]                                          # everything after the module docstring
    assert "ground_truth" not in code and "read_csv" not in code


# ── Prophet smoke test ─────────────────────────────────────────────────────────

def test_prophet_learns_an_event_lift_and_flags_an_unseen_event():
    pytest.importorskip("prophet")
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-10-01", "2026-01-31")
    cal = pd.DataFrame([
        dict(event="Boom", start_date="2025-03-01", end_date="2025-03-10", lead_in_days=0, dip_days=0, dip_factor=1.0),
        dict(event="Boom", start_date="2026-01-05", end_date="2026-01-14", lead_in_days=0, dip_days=0, dip_factor=1.0),
        dict(event="Boom", start_date="2026-04-01", end_date="2026-04-10", lead_in_days=0, dip_days=0, dip_factor=1.0),
        dict(event="Never", start_date="2026-05-01", end_date="2026-05-10", lead_in_days=0, dip_days=0, dip_factor=1.0),
    ])
    reg = qe.event_regressors(dates, cal)
    lam = 20 * (1 + 1.0 * reg["ev_boom_in"].to_numpy())             # a true +100% lift during Boom
    series = pd.DataFrame({"ds": dates, "y": rng.poisson(lam).astype(float)})
    fc = qe.fit_prophet(series, cal)
    assert fc is not None and fc.events_without_history == ["Never"]
    boom = fc.window_sum(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-10")) / 10
    plain = fc.window_sum(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-10")) / 10
    assert 1.7 < boom / plain < 2.3, f"learned lift {boom / plain:.2f}, true 2.0"


# ── Phase 3 additions: validated horizon + measured coverage ───────────────────

TABLE = {"overall": 0.90, "events_with_history": 0.94, "events_without_history": 0.78}


def test_coverage_estimate_depends_on_whether_the_events_had_history():
    assert qe.coverage_for([], TABLE)[0] == 0.94
    assert qe.coverage_for(["events_without_history:Ramadan"], TABLE)[0] == 0.78
    assert qe.coverage_for([], None)[0] is None


def test_no_coverage_is_claimed_beyond_the_validated_horizon():
    est, basis = qe.coverage_for(["beyond_validated_horizon", "events_without_history:Ramadan"], TABLE)
    assert est is None and "not measured" in basis


def test_order_quantity_flags_a_window_that_starts_beyond_the_validated_horizon():
    far = qe.CoverWindow(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-10"), "t")      # train_end 31 Jan: 60 days away
    r = run(window=far, coverage_table=TABLE)
    assert r.horizon_days == 60 and "beyond_validated_horizon" in r.flags and r.coverage_estimate is None
    near = run(coverage_table=TABLE)                                                        # window starts 11 Feb: 11 days
    assert near.horizon_days == 11 and "beyond_validated_horizon" not in near.flags and near.coverage_estimate == 0.94
