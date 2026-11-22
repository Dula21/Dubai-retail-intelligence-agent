"use client";
import { useMemo, useState } from "react";
import type { Recommendation, Status } from "@/lib/api";
import { URGENCY_LABEL, TRIGGER_LABEL, confidenceTier, day, num, pct } from "@/lib/format";

const TABS: { key: Status | "all"; label: string }[] = [
  { key: "pending", label: "Pending" },
  { key: "approved", label: "Approved" },
  { key: "modified", label: "Modified" },
  { key: "rejected", label: "Rejected" },
  { key: "all", label: "All" },
];

export default function RecList({
  recs, selectedId, onSelect,
}: { recs: Recommendation[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const [tab, setTab] = useState<Status | "all">("pending");
  const [q, setQ] = useState("");

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: recs.length, pending: 0, approved: 0, modified: 0, rejected: 0 };
    recs.forEach((r) => { c[r.status] += 1; });
    return c;
  }, [recs]);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return recs.filter((r) =>
      (tab === "all" || r.status === tab) &&
      (!needle || r.sku_id.toLowerCase().includes(needle) || r.product_name.toLowerCase().includes(needle) ||
        (r.event ?? "").toLowerCase().includes(needle)));
  }, [recs, tab, q]);

  return (
    <section className="card" aria-label="Recommendations">
      <div className="toolbar">
        <div className="tabs" role="group" aria-label="Filter by status">
          {TABS.map((t) => (
            <button key={t.key} className="tab" aria-pressed={tab === t.key} onClick={() => setTab(t.key)}>
              {t.label}<span>{counts[t.key]}</span>
            </button>
          ))}
        </div>
        <input className="search" placeholder="Search SKU, product or event" value={q}
               onChange={(e) => setQ(e.target.value)} aria-label="Search recommendations" />
      </div>
      {shown.length === 0 ? (
        <div className="empty">
          {recs.length === 0
            ? "No recommendations yet. Run a scan to raise some."
            : "Nothing matches this filter."}
        </div>
      ) : (
        <ul className="list" role="listbox" aria-label="Recommendation list">
          {shown.map((r) => (
            <li key={r.rec_id} role="presentation">
              <button className="item" role="option" aria-selected={r.rec_id === selectedId} onClick={() => onSelect(r.rec_id)}>
                <div className="item-top">
                  <span className={`pill ${r.urgency}`}>{URGENCY_LABEL[r.urgency]}</span>
                  <span className="item-title">{r.sku_id} · {r.product_name}</span>
                  {r.status !== "pending" && <span className={`pill ${r.status}`}>{r.status}</span>}
                </div>
                <div className="item-meta">
                  <span>Order <b>{num(r.status === "modified" && r.final_qty != null ? r.final_qty : r.order_qty)}</b> units
                    {r.status === "modified" && <> (was {num(r.order_qty)})</>}</span>
                  <span>By <b>{day(r.order_by_date)}</b></span>
                  <span>{r.event ? <>For <b>{r.event}</b></> : <b>{TRIGGER_LABEL[r.trigger] ?? r.trigger}</b>}</span>
                  <span className={`pill tier-${confidenceTier(r)}`} title={r.confidence_basis}>{pct(r.confidence)} confidence</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
