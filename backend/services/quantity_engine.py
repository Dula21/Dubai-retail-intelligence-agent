"""
services/quantity_engine.py
----------------------------
Deterministic ORDER-QUANTITY engine (Phase 3). Pure functions, no LLM, no agent imports.

  order_qty = ceil( max(0, forecast_over_window x (1 + safety_q) - projected_on_hand - open_orders) )

Every term is computed here and travels in a QuantityResult with its own trace lines, so the
recommendation node only EXPLAINS a number; it never invents one.

All data is SYNTHETIC (see data/generate_synthetic.py). Nothing here reads ground_truth_demand.csv:
a test (tests/test_quantity_engine.py) fails if any module under backend/ mentions it.

Engineering decisions (decision / alternative / why / trade-off), ready for the write-up
------------------------------------------------------------------------------------------
1. Censored demand: sales are not demand. On a stockout day customers wanted more than was sold, so
   fitting a forecaster to raw sales learns demand that is too LOW, worst on exactly the SKUs that
   sell out. The generator writes `stockout_occurred` (an out-of-stock flag a real retailer also has).
   Chosen: mask censored days as NaN (Prophet ignores NaN targets). Three modes are implemented so the
   choice is MEASURED, not assumed (eval/backtest_quantity.py --compare-masking):
     none  fit raw sales                        simplest, biased low
     flag  mask only flagged days               leaves the zero-demand days INSIDE a stockout run,
                                                 which are legitimate but a biased sample of that run
     run   mask whole stockout runs (default)   a run = consecutive days that are flagged or sold 0,
                                                 containing at least one flagged day
   Alternative: model the censoring explicitly (Tobit / censored likelihood): more accurate, far more
   machinery than SME scale justifies. Trade-off of masking: it discards data, so a SKU that is out of
   stock for most of an event learns little about that event (reported as `high_masked_share`).

2. Event effects: ONE learned coefficient per calendar event window (lead-in, in-event, post-event dip)
   passed to Prophet as multiplicative regressors, not Prophet's `holidays=` frame. Prophet's holidays
   make one coefficient per day-offset of the window (a 30-day Ramadan with 14 lead-in days = 45
   coefficients) and each SKU has only 1-2 past occurrences, so those are not identifiable. Trade-off:
   the effect is flat across the window (the real lead-in is a ramp). The calendar's dates come from
   the shared uae_retail_calendar.csv. The lift TABLE is deliberately NOT an input: the data was
   generated from it, so using it would make the backtest circular.

3. Safety stock: split-conformal multiplier on the window forecast. For past event windows we measure
   actual/forecast - 1, and take the finite-sample quantile at the target service level (default 0.90),
   separately for low / medium / high volume SKUs (Poisson noise is relatively larger for slow movers).
   Chosen over Prophet's parametric intervals: no distributional assumption, cheap (fit with
   uncertainty_samples=0) and the coverage is MEASURED on held-out windows, which is what the
   calibrated confidence needs. Trade-off: it needs a backtest history, and assumes future windows
   resemble past ones (exchangeability).

4. Stock netting is only as good as the stock number. The snapshot has a date; if it is older than
   MAX_SNAPSHOT_AGE_DAYS the result says so. Stock at the start of the cover window is PROJECTED as
   snapshot minus forecast demand, assuming no replenishment in between (conservative: it orders more).
   Open purchase orders are not in any dataset: they are 0 and flagged `open_orders_unknown`.
   Unknown stock (-1) gives NO quantity, never a guess (hard fallback for factual fields).

Scale ceiling: one Prophet fit per SKU is fine for hundreds of SKUs offline (batch); it is NOT a
per-request operation. At noon scale this becomes hierarchical / global models trained in a scheduled job.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ── Constants (set once, BEFORE looking at any backtest result) ────────────────
REVIEW_PERIOD_DAYS = 14        # non-event orders cover lead time + this many days
ORDER_BUFFER_DAYS = 5          # same buffer the seasonality node uses for the order-by date
DEFAULT_SERVICE_LEVEL = 0.90
MAX_SNAPSHOT_AGE_DAYS = 7
MIN_TRAIN_DAYS = 300           # fewer observed days than this -> no forecast
HIGH_MASKED_SHARE = 0.25       # more than this share of days masked -> flag
FORECAST_FLOOR_UNITS = 1.0     # relative errors use max(forecast, this) to avoid dividing by ~0
VALIDATED_HORIZON_DAYS = 30    # backtest order dates were at most lead(21)+buffer(5) days before the window start
MIN_PER_BIN = 30               # fewer calibration cases than this in a volume bin -> use the global quantile
EVENT_PRIOR_SCALE = 2.0        # prior on an event coefficient (relative lift); not tuned on test results

MASK_MODES = ("none", "flag", "run")


def slug(name: str) -> str:
    """'Eid al-Fitr' -> 'eid_al_fitr' (same rule as sales_node._slug)."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# ── 1. Censoring ───────────────────────────────────────────────────────────────

def censored_days(units: pd.Series, stockout: pd.Series, mode: str = "run") -> pd.Series:
    """
    Boolean Series (True = observed sales are NOT a demand observation), aligned to `units`' index.

    `units` and `stockout` must be sorted by date, one row per day.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask mode must be one of {MASK_MODES}, got {mode!r}")
    flag = stockout.fillna(0).astype(int) == 1
    if mode == "none":
        return pd.Series(False, index=units.index)
    if mode == "flag":
        return flag
    # "run": consecutive days that are flagged OR sold nothing; a run is censored when it holds a flagged day.
    suspect = flag | (units.fillna(0) == 0)
    run_id = (suspect != suspect.shift()).cumsum()
    run_has_flag = flag.groupby(run_id).transform("any")
    return suspect & run_has_flag


def demand_series(sku_sales: pd.DataFrame, mode: str = "run", cutoff: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Prophet-ready frame (ds, y) for ONE SKU: observed sales with censored days set to NaN.
    sku_sales needs columns date, units_sold and (optionally) stockout_occurred. Rows after `cutoff` are dropped,
    which is how a backtest keeps the future out of the fit.
    """
    df = sku_sales.sort_values("date")
    if cutoff is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff)]
    stock = df["stockout_occurred"] if "stockout_occurred" in df else pd.Series(0, index=df.index)
    cens = censored_days(df["units_sold"].reset_index(drop=True), stock.reset_index(drop=True), mode).to_numpy()
    y = df["units_sold"].astype(float).to_numpy().copy()
    y[cens] = np.nan
    return pd.DataFrame({"ds": df["date"].to_numpy(), "y": y})


# ── 2. Calendar -> regressors ──────────────────────────────────────────────────

def event_regressors(dates: Iterable, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    0/1 columns per event window, indexed like `dates`:
      ev_<slug>_lead  the lead_in_days before the event starts (demand builds)
      ev_<slug>_in    the event itself
      ev_<slug>_dip   dip_days after the event (only for events with dip_factor < 1)
    All occurrences of one event share a column, so a coefficient is learned from every past occurrence.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    cols: dict[str, np.ndarray] = {}

    def put(name: str, mask: np.ndarray) -> None:
        cols[name] = np.maximum(cols.get(name, np.zeros(len(idx))), mask.astype(float))

    for r in calendar.itertuples(index=False):
        start, end = pd.Timestamp(r.start_date), pd.Timestamp(r.end_date)
        lead = int(getattr(r, "lead_in_days", 0) or 0)
        dip = int(getattr(r, "dip_days", 0) or 0)
        dip_factor = float(getattr(r, "dip_factor", 1.0) if pd.notna(getattr(r, "dip_factor", 1.0)) else 1.0)
        s = slug(r.event)
        put(f"ev_{s}_in", np.asarray((idx >= start) & (idx <= end)))
        if lead > 0:
            put(f"ev_{s}_lead", np.asarray((idx >= start - pd.Timedelta(days=lead)) & (idx < start)))
        if dip > 0 and dip_factor < 1.0:
            put(f"ev_{s}_dip", np.asarray((idx > end) & (idx <= end + pd.Timedelta(days=dip))))
    return pd.DataFrame(cols, index=idx)


def event_names_in_window(calendar: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Calendar events whose lead-in..event span overlaps [start, end]."""
    names: list[str] = []
    for r in calendar.itertuples(index=False):
        lo = pd.Timestamp(r.start_date) - pd.Timedelta(days=int(getattr(r, "lead_in_days", 0) or 0))
        hi = pd.Timestamp(r.end_date)
        if lo <= end and hi >= start and r.event not in names:
            names.append(r.event)
    return names


# ── 3. Forecasters ─────────────────────────────────────────────────────────────

@dataclass
class Forecast:
    """A fitted demand model for ONE SKU. predict() returns non-negative units per day."""
    method: str
    train_end: pd.Timestamp
    n_train_days: int                    # days that carried a demand observation
    masked_share: float                  # share of history days masked as censored
    events_without_history: list[str] = field(default_factory=list)
    _predict: object = None              # callable(DatetimeIndex) -> ndarray

    def predict(self, dates: Iterable) -> np.ndarray:
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        if len(idx) == 0:
            return np.zeros(0)
        return np.maximum(np.asarray(self._predict(idx), dtype=float), 0.0)

    def window_sum(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        if end < start:
            return 0.0
        return float(self.predict(pd.date_range(start, end)).sum())


def fit_prophet(series: pd.DataFrame, calendar: pd.DataFrame, mask_mode: str = "run") -> Optional[Forecast]:
    """
    Fit Prophet on (ds, y with NaN for censored days) + one learned coefficient per calendar event window.
    Returns None when there is too little uncensored history (the caller reports 'insufficient data').
    """
    from prophet import Prophet  # imported here so the module loads (and tests run) without the heavy install

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    obs = series["y"].notna()
    n_obs = int(obs.sum())
    if n_obs < MIN_TRAIN_DAYS:
        return None

    train = series.copy()
    train["ds"] = pd.to_datetime(train["ds"])
    regs = event_regressors(train["ds"], calendar)
    seen = [c for c in regs.columns if regs.loc[obs.to_numpy(), c].sum() > 0]      # windows with >=1 observed day
    unseen_names = _events_from_columns(calendar, [c for c in regs.columns if c not in seen])
    for c in seen:
        train[c] = regs[c].to_numpy()

    m = Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", uncertainty_samples=0)
    for c in seen:
        m.add_regressor(c, mode="multiplicative", prior_scale=EVENT_PRIOR_SCALE, standardize=False)
    m.fit(train)

    def _predict(idx: pd.DatetimeIndex) -> np.ndarray:
        future = pd.DataFrame({"ds": idx})
        r = event_regressors(idx, calendar)
        for c in seen:
            future[c] = r[c].to_numpy() if c in r.columns else 0.0
        return m.predict(future)["yhat"].to_numpy()

    return Forecast(
        method=f"prophet[{mask_mode}]", train_end=train["ds"].max(), n_train_days=n_obs,
        masked_share=float(1 - n_obs / len(series)), events_without_history=unseen_names, _predict=_predict,
    )


def _events_from_columns(calendar: pd.DataFrame, columns: list[str]) -> list[str]:
    """Map regressor column names back to calendar event names."""
    by_slug = {slug(e): e for e in calendar["event"].unique()}
    out: list[str] = []
    for c in columns:
        m = re.match(r"ev_(.+)_(lead|in|dip)$", c)
        name = by_slug.get(m.group(1)) if m else None
        if name and name not in out:
            out.append(name)
    return sorted(out)


def fit_naive(series: pd.DataFrame, window_days: int = 28) -> Optional[Forecast]:
    """Baseline: flat forecast = mean of the uncensored days in the last `window_days` days."""
    tail = series.tail(window_days)["y"].dropna()
    if len(tail) < 7:
        return None
    level = float(tail.mean())
    n_obs = int(series["y"].notna().sum())
    return Forecast(method="naive_trailing_mean", train_end=pd.to_datetime(series["ds"]).max(), n_train_days=n_obs,
                    masked_share=float(1 - n_obs / len(series)), _predict=lambda idx: np.full(len(idx), level))


# ── 4. Safety stock: split-conformal multiplier ────────────────────────────────

def relative_errors(forecasts: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """(actual - forecast) / max(forecast, FORECAST_FLOOR_UNITS). Positive = demand beat the forecast."""
    f = np.asarray(forecasts, dtype=float)
    a = np.asarray(actuals, dtype=float)
    return (a - f) / np.maximum(f, FORECAST_FLOOR_UNITS)


def conformal_quantile(errors: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)*level)-th smallest error (max if that exceeds n)."""
    e = np.sort(np.asarray(errors, dtype=float))
    n = len(e)
    if n == 0:
        raise ValueError("no calibration errors")
    k = min(math.ceil((n + 1) * level), n)
    return float(e[k - 1])


@dataclass(frozen=True)
class SafetyMultiplier:
    """
    target = forecast x multiplier(forecast). One quantile per volume bin (low / medium / high forecast
    total), because Poisson noise is relatively larger for slow movers. Bins with fewer than MIN_PER_BIN
    calibration cases use the global quantile.
    """
    service_level: float
    edges: tuple            # upper edges of all bins but the last (forecast units)
    q: tuple                # one relative-error quantile per bin
    n: tuple                # calibration cases per bin
    global_q: float
    n_total: int

    @classmethod
    def fit(cls, forecasts: np.ndarray, actuals: np.ndarray, service_level: float = DEFAULT_SERVICE_LEVEL,
            n_bins: int = 3) -> "SafetyMultiplier":
        f = np.asarray(forecasts, dtype=float)
        err = relative_errors(f, actuals)
        gq = conformal_quantile(err, service_level)
        edges = tuple(float(x) for x in np.quantile(f, [i / n_bins for i in range(1, n_bins)])) if n_bins > 1 else ()
        bin_ids = np.digitize(f, edges)
        qs, ns = [], []
        for b in range(len(edges) + 1):
            e_b = err[bin_ids == b]
            ns.append(int(len(e_b)))
            qs.append(conformal_quantile(e_b, service_level) if len(e_b) >= MIN_PER_BIN else gq)
        return cls(service_level, edges, tuple(qs), tuple(ns), gq, int(len(err)))

    def q_for(self, forecast_units: float) -> float:
        return self.q[int(np.digitize([forecast_units], self.edges)[0])]

    def multiplier(self, forecast_units: float) -> float:
        return 1.0 + max(self.q_for(forecast_units), -0.5)

    def to_dict(self) -> dict:
        return {"service_level": self.service_level, "edges": list(self.edges), "q": list(self.q),
                "n": list(self.n), "global_q": self.global_q, "n_total": self.n_total}

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyMultiplier":
        return cls(d["service_level"], tuple(d["edges"]), tuple(d["q"]), tuple(d["n"]), d["global_q"], d["n_total"])


# ── 5. Cover window + the quantity ─────────────────────────────────────────────

@dataclass(frozen=True)
class CoverWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str


def cover_window(as_of: date, lead_days: int, event: Optional[dict] = None,
                 review_days: int = REVIEW_PERIOD_DAYS) -> Optional[CoverWindow]:
    """
    The demand an order placed on `as_of` must cover.
      event given {start, end, lead_in_days}: from the later of (arrival, start of the event's lead-in) to the event end.
      no event: from arrival to arrival + review_days.
    None when the event is already over, or an order placed on `as_of` would arrive after it ends.
    """
    as_of_ts = pd.Timestamp(as_of)
    arrival = as_of_ts + pd.Timedelta(days=int(lead_days))
    if event is None:
        return CoverWindow(arrival, arrival + pd.Timedelta(days=review_days - 1),
                           f"arrival (+{lead_days}d) plus {review_days} days of cover")
    start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
    if end < as_of_ts:
        return None
    lead_in_start = start - pd.Timedelta(days=int(event.get("lead_in_days", 0) or 0))
    w_start = max(arrival, lead_in_start, as_of_ts + pd.Timedelta(days=1))
    if w_start > end:                       # the order would arrive after the event is over
        return None
    return CoverWindow(w_start, end, "event lead-in start (or arrival if later) to event end")


@dataclass
class QuantityResult:
    sku_id: str
    as_of: str
    method: str
    order_qty: Optional[int]                 # None = insufficient data (never a guess)
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    forecast_units: Optional[float] = None
    safety_multiplier: Optional[float] = None
    target_units: Optional[float] = None
    on_hand_snapshot: Optional[int] = None
    snapshot_date: Optional[str] = None
    snapshot_age_days: Optional[int] = None
    projected_on_hand: Optional[float] = None
    open_orders: int = 0
    horizon_days: Optional[int] = None          # days from the last training day to the window start
    coverage_estimate: Optional[float] = None   # measured share of held-out windows the target covered; None = not validated
    coverage_basis: str = ""
    flags: list = field(default_factory=list)
    trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def coverage_for(flags: list, table: Optional[dict]) -> tuple[Optional[float], str]:
    """
    Calibrated confidence: the share of HELD-OUT backtest windows that the target covered, for the group this
    recommendation belongs to. None when the recommendation is outside what the backtest validated.
    `table` = {"events_with_history": x, "events_without_history": y, "overall": z, ...} from data/quantity_calibration.json.
    """
    if not table:
        return None, "no calibration table"
    if "beyond_validated_horizon" in flags:
        return None, "window starts beyond the horizon the backtest validated; coverage not measured here"
    if "non_event_window" in flags:
        return None, "window has no event; every backtest case was an event window, so coverage is not measured here"
    if any(f.startswith("events_without_history") for f in flags):
        return table.get("events_without_history"), "held-out windows overlapping an event with no learnable history"
    return table.get("events_with_history"), "held-out windows whose events all had history"


def order_quantity(
    sku_id: str,
    forecast: Optional[Forecast],
    window: Optional[CoverWindow],
    safety: Optional[SafetyMultiplier],
    on_hand: Optional[float],
    snapshot_date: Optional[date],
    as_of: date,
    open_orders: Optional[int] = None,
    calendar: Optional[pd.DataFrame] = None,
    coverage_table: Optional[dict] = None,
    validated_horizon_days: int = VALIDATED_HORIZON_DAYS,
) -> QuantityResult:
    """
    The arithmetic, with every input and caveat recorded.

      target   = window forecast x safety multiplier
      projected_on_hand = snapshot stock, minus forecast demand from the snapshot to the window start
                          (floored at 0, no replenishment assumed in between)
      qty      = ceil(max(0, target - projected_on_hand - open_orders))
    """
    res = QuantityResult(sku_id=sku_id, as_of=as_of.isoformat(), method=forecast.method if forecast else "none",
                         order_qty=None)

    def stop(flag: str, line: str) -> QuantityResult:
        res.flags.append(flag)
        res.trace.append(line)
        return res

    if window is None:
        return stop("no_cover_window", f"{sku_id}: the event is over, or an order placed now would arrive after it ends; no quantity")
    res.window_start, res.window_end = window.start.date().isoformat(), window.end.date().isoformat()
    if forecast is None:
        return stop("insufficient_history", f"{sku_id}: not enough uncensored sales history to forecast; no quantity")
    if safety is None:
        return stop("no_calibration", f"{sku_id}: no calibrated safety multiplier available; no quantity")
    if on_hand is None or on_hand < 0:
        return stop("stock_unknown", f"{sku_id}: current stock unknown (-1 or missing); no quantity")

    f_window = forecast.window_sum(window.start, window.end)
    mult = safety.multiplier(f_window)
    target = f_window * mult
    res.forecast_units, res.safety_multiplier, res.target_units = round(f_window, 1), round(mult, 3), round(target, 1)
    res.on_hand_snapshot = int(on_hand)

    snap = snapshot_date or as_of
    res.snapshot_date = snap.isoformat()
    res.snapshot_age_days = (as_of - snap).days
    if res.snapshot_age_days > MAX_SNAPSHOT_AGE_DAYS:
        res.flags.append("stock_snapshot_stale")

    # Project stock to the start of the window, assuming nothing arrives in between.
    snap_ts = pd.Timestamp(snap)
    projected = float(on_hand)
    if window.start > snap_ts + pd.Timedelta(days=1):
        depletion = forecast.window_sum(snap_ts + pd.Timedelta(days=1), window.start - pd.Timedelta(days=1))
        projected = max(0.0, on_hand - depletion)
        res.flags.append("stock_projected_without_replenishment")
    res.projected_on_hand = round(projected, 1)

    if open_orders is None:
        res.flags.append("open_orders_unknown")
        oo = 0
    else:
        oo = int(open_orders)
    res.open_orders = oo

    if calendar is not None and forecast.events_without_history:
        overlap = [e for e in event_names_in_window(calendar, window.start, window.end)
                   if e in forecast.events_without_history]
        if overlap:
            res.flags.append("events_without_history:" + ",".join(overlap))
    if forecast.masked_share > HIGH_MASKED_SHARE:
        res.flags.append("high_masked_share")

    res.horizon_days = int((window.start - forecast.train_end).days)
    if res.horizon_days > validated_horizon_days:
        res.flags.append("beyond_validated_horizon")
    if window is not None and window.reason.startswith("arrival"):      # the generic lead-time + review window
        res.flags.append("non_event_window")
    res.coverage_estimate, res.coverage_basis = coverage_for(res.flags, coverage_table)

    res.order_qty = int(math.ceil(max(0.0, target - projected - oo)))
    res.trace.append(
        f"{sku_id}: window {res.window_start} to {res.window_end}; forecast {res.forecast_units} units "
        f"x safety {res.safety_multiplier} (service level {safety.service_level:.0%}) = target {res.target_units}; "
        f"stock {res.on_hand_snapshot} on {res.snapshot_date} (age {res.snapshot_age_days}d) -> "
        f"projected {res.projected_on_hand} at window start; open orders {oo}"
        f"{' (unknown, assumed 0)' if open_orders is None else ''}; order {res.order_qty} units; "
        + (f"coverage estimate {res.coverage_estimate:.0%} ({res.coverage_basis})" if res.coverage_estimate is not None
           else f"coverage estimate: none ({res.coverage_basis})")
        + (f" [flags: {', '.join(res.flags)}]" if res.flags else "")
    )
    return res
