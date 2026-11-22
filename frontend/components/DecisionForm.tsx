"use client";
import { useEffect, useState } from "react";
import { api, Recommendation } from "@/lib/api";
import { num } from "@/lib/format";

type Kind = "approve" | "modify" | "reject";

export default function DecisionForm({ rec, onDecided }: { rec: Recommendation; onDecided: () => void }) {
  const [kind, setKind] = useState<Kind>("approve");
  const [actor, setActor] = useState("");
  const [reason, setReason] = useState("");
  const [qty, setQty] = useState(String(rec.order_qty));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { try { setActor(localStorage.getItem("dri-actor") ?? ""); } catch { /* private mode */ } }, []);
  useEffect(() => { setKind("approve"); setReason(""); setQty(String(rec.order_qty)); setError(null); }, [rec.rec_id, rec.order_qty]);

  const n = Number(qty);
  const qtyValid = qty.trim() !== "" && Number.isInteger(n) && n >= 0 && n !== rec.order_qty;
  const delta = qtyValid && rec.order_qty ? ((n - rec.order_qty) / rec.order_qty) * 100 : null;
  const needsReason = kind !== "approve";
  const ready = actor.trim().length > 0 && (!needsReason || reason.trim().length >= 3) && (kind !== "modify" || qtyValid);

  async function submit() {
    setBusy(true); setError(null);
    try {
      try { localStorage.setItem("dri-actor", actor.trim()); } catch { /* ignore */ }
      await api.decide(rec.rec_id, {
        decision: kind, actor: actor.trim(),
        ...(needsReason ? { reason: reason.trim() } : {}),
        ...(kind === "modify" ? { quantity: n } : {}),
      });
      onDecided();
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  return (
    <div className="decide">
      <h2 style={{ marginBottom: 8 }}>Your decision</h2>
      <div className="seg" role="group" aria-label="Decision">
        {(["approve", "modify", "reject"] as Kind[]).map((k) => (
          <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)}>{k[0].toUpperCase() + k.slice(1)}</button>
        ))}
      </div>
      <p className="hint">
        {kind === "approve" && <>Order the system&apos;s quantity: <b>{num(rec.order_qty)} units</b>.</>}
        {kind === "modify" && "Order a different quantity. The agent's own number is kept alongside yours."}
        {kind === "reject" && "Do not order. The recommendation stays on record with your reason."}
      </p>
      <div className="row" style={{ marginTop: 10 }}>
        <label className="field">Your name
          <input value={actor} onChange={(e) => setActor(e.target.value)} maxLength={80} placeholder="e.g. Dulasi" />
        </label>
        {kind === "modify" && (
          <label className="field">New quantity
            <input type="number" min={0} value={qty} onChange={(e) => setQty(e.target.value)} style={{ width: 110 }} />
          </label>
        )}
      </div>
      {kind === "modify" && qty !== String(rec.order_qty) && (
        <p className="hint">
          {qtyValid
            ? <>{num(rec.order_qty)} → <b>{num(n)}</b> ({delta! >= 0 ? "+" : ""}{delta!.toFixed(1)}%)</>
            : "Enter a whole number different from the recommended quantity."}
        </p>
      )}
      {needsReason && (
        <label className="field" style={{ marginTop: 10 }}>Reason (required)
          <textarea value={reason} onChange={(e) => setReason(e.target.value)} maxLength={1000}
                    placeholder={kind === "modify" ? "e.g. Supplier sells in full pallets of 200" : "e.g. Already overstocked at the warehouse"} />
        </label>
      )}
      {error && <div className="notice error" role="alert" style={{ marginTop: 10, marginBottom: 0 }}>{error}</div>}
      <div style={{ marginTop: 12 }}>
        <button className={`btn ${kind === "reject" ? "danger" : "primary"}`} onClick={submit} disabled={!ready || busy}>
          {busy ? "Saving…" : kind === "approve" ? "Approve order" : kind === "modify" ? "Save modified order" : "Reject"}
        </button>
        <span className="hint" style={{ marginLeft: 10 }}>A decision is final.</span>
      </div>
    </div>
  );
}
