import type { Recommendation, Urgency } from "./api";

/** Plain-language caveats. Same wording the backend gives the model, so the screen and the chat agree. */
const FLAG_TEXT: Record<string, string> = {
  stock_snapshot_stale: "The stock figure is out of date.",
  stock_projected_without_replenishment: "Stock was projected forward assuming no deliveries arrive.",
  open_orders_unknown: "Open purchase orders are unknown and assumed to be zero.",
  high_masked_share: "A large share of this product's history was lost to stock-outs, so the forecast learned less.",
  beyond_validated_horizon: "The window is farther out than the backtest validated.",
  lead_time_assumed: "The supplier lead time is assumed, not recorded.",
  non_event_window: "This window has no event, and the backtest only validated event windows.",
};

export function flagText(flag: string): string {
  if (flag.startsWith("events_without_history:"))
    return `Overlaps an event with no sales history to learn from (${flag.split(":")[1]}).`;
  return FLAG_TEXT[flag] ?? flag.replace(/_/g, " ");
}

export const URGENCY_LABEL: Record<Urgency, string> = { late: "Late", now: "Order today", upcoming: "Upcoming" };

export function pct(x: number | null | undefined, digits = 0): string {
  return x == null ? "n/a" : `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 0): string {
  return x == null ? "n/a" : x.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function day(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const d = new Date(iso.length <= 10 ? `${iso}T00:00:00` : iso);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  return new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/** Confidence is the measured coverage of a group of past windows, or a fixed low cap when not validated. */
export function confidenceTier(r: Pick<Recommendation, "confidence">): "high" | "mid" | "low" {
  return r.confidence >= 0.85 ? "high" : r.confidence >= 0.6 ? "mid" : "low";
}

export const TRIGGER_LABEL: Record<string, string> = {
  runway: "Stock runway",
  event: "Upcoming event",
  "runway+event": "Runway + event",
};
