# Phase 3 results: reorder quantity engine

**All data is synthetic.** Demand lifts come from `event_lifts.csv`, so these numbers show that the pipeline works, not how real
Dubai demand behaves. Reproduce with:

```
docker compose exec api python eval/backtest_quantity.py --sample-skus 100 --workers 2 --compare-masking
```
(about 20 minutes on 2 workers; writes `eval/quantity_backtest.json` and `eval/quantity_backtest.cases.csv`, both committed).
The figures below were re-run and checked against those files on 2026-11-22 (1,500 cases, split 2026-01-09).

## The claim for "reorder recommendation precision"

The project target was "precision above 80%". That wording does not say what a correct recommendation is, so it was fixed
before measuring, and the honest answer turned out to be two numbers, not one:

> At a 90% service-level target, the order quantity covered actual demand in **89.9%** of 526 held-out cases on SKUs the event lifts
> (SKU x past event, time-split: calibrated on windows that ended before 2026-01-09, tested after). For events the model had
> history for, coverage was about 94%; for windows overlapping an event with no history it was 78.2% (n=147).
> When the target covered demand, it overshot by a median of 31.9%.

**What is not claimed.** Under the strict joint definition (no stockout AND overshoot within 30%) the engine scores **41%**,
and 66% if 50% overshoot is tolerated. The "above 80%" goal is therefore **not met** under that definition. The two ways to
raise it are tighter forecasts (the held-out WAPE is 12.1%) or a lower service level, which trades stock-outs for cash tied up
in stock; neither is a free fix. The decision was to report coverage as the headline, because avoiding stock-outs is what a
reorder recommendation is for, and to publish the overshoot beside it.

Interview version: "I define a recommendation as correct if it covers demand. That held in 89.9% of 526 out-of-sample cases on lifted SKUs
against a 90% target, so the calibration is honest. The cost is overshoot: a median 32% when covered. The stricter 'covered
and not wasteful' measure is 41%, and I report that too."

## Measured results (100-SKU sample of the 500-SKU dataset, 1,500 cases: 475 for calibration, 760 for testing, of which 526 are on SKUs the event lifts)

| Measure | Value |
|---|---|
| Coverage at 90% target, lifted SKUs (headline) | 89.9% (n=526) |
| Coverage at 90% target, all 760 test cases | 91.2% (strict precision 41.6%) |
| Coverage, events with history | about 94% (derived from totals, not measured directly) |
| Coverage, events without history | 78.2% (n=147) |
| Median overshoot when covered | 31.9% |
| Strict precision (covered and overshoot at most 30%) | 41% (66% at 50% tolerance) |
| Forecast WAPE / raw bias | 12.1% / -6.6% (the safety multiplier compensates) |
| Masking: none / flagged days / whole stockout runs | WAPE 28.8% / 12.4% / 11.9% |

Coverage by category (n): beauty 83.9% (112), electronics 92.3% (65), fashion 89.8% (166), food 96.8% (93), home 88.9% (90).
Beauty is about 1.7 standard errors below target; noted, not corrected.

## Known limits (state these before an interviewer finds them)

- **Synthetic data**, generated from the same lift table the calendar uses. The engine deliberately does not read that table
  (a test enforces it), but the data's structure is simple, which flatters every number above.
- **Event windows only.** Every backtest case was an event window. Quantities for windows with no event, or farther out than 30
  days, carry no coverage claim and a fixed confidence of 0.4.
- **Overlapping events** (Eid al-Fitr and Mother's Day in 2026) are not separated: biases of -10% and -9%.
- **Confidence is group coverage, not a per-SKU probability.** It says how often targets of this kind covered demand in past
  held-out windows.
- **The lift-table comparator** in the backtest (+49% overshoot) is my own implementation and likely mis-specified; do not quote it.
- **Stock figures** have no timestamp in the dataset; the snapshot is treated as dated at the last sales date, so answers given
  later are flagged stale. Open purchase orders are unknown and assumed zero.
- **Scale:** one Prophet fit per SKU, lazily cached; fine for hundreds to low thousands of SKUs. Next architectural step at
  scale: batch forecasts written to PostgreSQL, request path reads only.
