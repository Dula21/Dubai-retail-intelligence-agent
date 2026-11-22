"use client";
import { useState } from "react";
import { api, AskResponse } from "@/lib/api";
import Rich from "@/components/Rich";
import { confidenceTier, day, flagText, num, pct } from "@/lib/format";

const EXAMPLES = [
  "What should I stock more of before Ramadan?",
  "What should I reorder before DSF?",
  "Show me all summer dresses under AED 200",
  "ما هي المنتجات التي يجب أن أعيد طلبها قبل رمضان؟",
];

export default function AskPage() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ask(text: string) {
    if (text.trim().length < 3) return;
    setBusy(true); setError(null); setRes(null);
    try { setRes(await api.ask(text.trim())); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  return (
    <>
      <section className="card card-pad" style={{ marginBottom: 14 }}>
        <h1>Ask the agent</h1>
        <p className="sub">Ask in English or Arabic. The agent checks the catalogue, sales, stock and the UAE calendar, then explains.
          Quantities come from the forecasting engine, never from the language model.</p>
        <form className="row" style={{ marginTop: 12 }} onSubmit={(e) => { e.preventDefault(); ask(q); }}>
          <label className="field" style={{ flex: 1, minWidth: 240 }}>Your question
            <input value={q} onChange={(e) => setQ(e.target.value)} maxLength={500} placeholder="What should I stock more of before Ramadan?" />
          </label>
          <button className="btn primary" type="submit" disabled={busy || q.trim().length < 3}>
            {busy ? <><span className="spinner" />Thinking…</> : "Ask"}
          </button>
        </form>
        <div className="examples">
          {EXAMPLES.map((e) => <button key={e} className="btn sm" onClick={() => { setQ(e); ask(e); }} disabled={busy}>{e}</button>)}
        </div>
      </section>

      {error && <div className="notice error" role="alert">{error}</div>}

      {res && (
        <div className="layout">
          <section className="card card-pad" aria-label="Answer">
            <div className="item-top" style={{ marginBottom: 8 }}>
              <span className={`pill tier-${confidenceTier({ confidence: res.confidence })}`}>{pct(res.confidence)} confidence</span>
              <span className="pill pending">{res.intent.replace(/_/g, " ")}</span>
              {res.fallback_triggered && <span className="pill now">insufficient data</span>}
            </div>
            <div dir={res.query_language === "ar" ? "rtl" : "ltr"}><Rich text={res.recommendation} /></div>
            {res.order_recommendations && res.order_recommendations.length > 0 && (
              <>
                <h3>Order quantities (from the forecasting engine)</h3>
                <div style={{ overflowX: "auto" }}>
                  <table className="t">
                    <thead><tr><th>SKU</th><th>Quantity</th><th>Window</th><th>Order by</th><th>Coverage</th></tr></thead>
                    <tbody>
                      {res.order_recommendations.map((o) => (
                        <tr key={o.sku_id} title={o.flags.map(flagText).join(" ")}>
                          <td>{o.sku_id}</td>
                          <td>{num(o.order_qty)}{o.indicative_only && <> <span className="pill now">indicative</span></>}</td>
                          <td>{day(o.window_start)} – {day(o.window_end)}</td>
                          <td>{day(o.order_by_date)}</td>
                          <td>{o.coverage_estimate == null ? "not validated" : pct(o.coverage_estimate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
          <section className="card card-pad" aria-label="Reasoning trace">
            <h2>Reasoning trace</h2>
            <ol className="trace">{res.reasoning_trace.map((l, i) => <li key={i}>{l}</li>)}</ol>
          </section>
        </div>
      )}
    </>
  );
}
