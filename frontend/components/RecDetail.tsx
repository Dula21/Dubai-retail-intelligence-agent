"use client";
import type { HistoryEvent, Recommendation } from "@/lib/api";
import { URGENCY_LABEL, TRIGGER_LABEL, confidenceTier, day, flagText, num, pct, when } from "@/lib/format";
import DecisionForm from "./DecisionForm";

const EVENT_LABEL: Record<string, string> = {
  created: "Raised by the agent", approved: "Approved", modified: "Modified", rejected: "Rejected",
};

export default function RecDetail({
  rec, history, onDecided,
}: { rec: Recommendation; history: HistoryEvent[]; onDecided: () => void }) {
  const decided = rec.status !== "pending";
  return (
    <article className="card detail" aria-label={`Recommendation for ${rec.sku_id}`}>
      <div className="detail-head">
        <div className="item-top">
          <span className={`pill ${rec.urgency}`}>{URGENCY_LABEL[rec.urgency]}</span>
          <span className={`pill ${rec.status}`}>{rec.status}</span>
          <span className="pill pending">{TRIGGER_LABEL[rec.trigger] ?? rec.trigger}</span>
          {rec.replay && <span className="pill pending">replay</span>}
        </div>
        <h1 style={{ marginTop: 8 }}>{rec.sku_id} · {rec.product_name}</h1>
        <p>{rec.headline}</p>
      </div>
      <div className="detail-body">
        {decided ? (
          <div className={`decided ${rec.status === "rejected" ? "rejected" : ""}`} style={{ marginTop: 12 }}>
            <strong style={{ textTransform: "capitalize" }}>{rec.status}</strong> by {rec.decided_by} on {when(rec.decided_at)}
            {rec.status === "rejected"
              ? <> — nothing ordered.</>
              : <> — order <b>{num(rec.final_qty)}</b> units
                  {rec.status === "modified" && <> (the agent recommended {num(rec.order_qty)}{rec.modification_delta_pct != null && <>, {rec.modification_delta_pct >= 0 ? "+" : ""}{rec.modification_delta_pct}%</>})</>}.</>}
            {rec.decision_reason && <div className="hint" style={{ color: "inherit" }}>Reason: {rec.decision_reason}</div>}
          </div>
        ) : (
          <div style={{ marginTop: 12 }}><DecisionForm rec={rec} onDecided={onDecided} /></div>
        )}

        <h3>The numbers</h3>
        <dl className="grid" style={{ margin: 0 }}>
          <div className="kv"><dt>Recommended quantity</dt><dd>{num(rec.order_qty)} <small>units</small></dd></div>
          <div className="kv"><dt>Order by</dt><dd>{day(rec.order_by_date)}</dd></div>
          <div className="kv"><dt>Cover window</dt><dd>{day(rec.window_start)} – {day(rec.window_end)}</dd></div>
          <div className="kv"><dt>Forecast demand</dt><dd>{num(rec.forecast_units, 1)} <small>units</small></dd></div>
          <div className="kv"><dt>Safety multiplier</dt><dd>×{num(rec.safety_multiplier, 3)}</dd></div>
          <div className="kv"><dt>Target stock</dt><dd>{num(rec.target_units, 1)} <small>units</small></dd></div>
          <div className="kv"><dt>Stock on hand</dt><dd>{num(rec.on_hand_snapshot)} <small>{rec.snapshot_age_days != null ? `(${rec.snapshot_age_days}d old)` : ""}</small></dd></div>
          <div className="kv"><dt>Open orders</dt><dd>{num(rec.open_orders)} <small>(unknown)</small></dd></div>
          <div className="kv"><dt>Confidence</dt>
            <dd><span className={`pill tier-${confidenceTier(rec)}`}>{pct(rec.confidence)}</span></dd></div>
        </dl>
        <p className="hint">Confidence basis: {rec.confidence_basis}. It is the share of past held-out windows that a target of this kind covered, not a probability for this SKU.</p>

        {rec.flags.length > 0 && (
          <>
            <h3>What to be aware of</h3>
            <ul className="caveats">{rec.flags.map((f) => <li key={f}>{flagText(f)}</li>)}</ul>
          </>
        )}

        <h3>Why the agent recommends this</h3>
        <ol className="trace">{rec.reasoning_trace.map((line, i) => <li key={i}>{line}</li>)}</ol>

        <h3>History</h3>
        <ul className="timeline">
          {history.length === 0 && <li><small>No history recorded.</small></li>}
          {history.map((h, i) => (
            <li key={i}>
              <b>{EVENT_LABEL[h.event_type] ?? h.event_type}</b> · {h.actor.startsWith("system:") ? "the agent" : h.actor}{" "}
              <small>{when(h.created_at)}</small>
              {h.event_type === "modified" && <div><small>{num(h.qty_before)} → {num(h.qty_after)} units</small></div>}
              {h.event_type !== "created" && h.reason && <div><small>Reason: {h.reason}</small></div>}
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}
