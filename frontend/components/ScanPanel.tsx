"use client";
import { useState } from "react";
import { api, ScanSummary } from "@/lib/api";
import { day, when } from "@/lib/format";

const NOT_ASSESSED_LABEL: Record<string, string> = {
  snapshot_stale: "stock figure too old",
  stock_unknown: "stock unknown",
  rate_unknown: "no usable sales rate",
};

export default function ScanPanel({ last, onDone }: { last: ScanSummary | null; onDone: () => void }) {
  const [asOf, setAsOf] = useState("");
  const [limit, setLimit] = useState("40");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(replayDate?: string) {
    setBusy(true);
    setError(null);
    const date = replayDate ?? asOf;
    try {
      await api.scan({ ...(date ? { as_of: date } : {}), limit: Number(limit) || 40 });
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const stale = last && !last.replay && (last.snapshot_age_days ?? 0) > 7;
  const notAssessed = last ? Object.entries(last.not_assessed).filter(([, n]) => n > 0) : [];

  return (
    <section className="card card-pad scan" aria-label="Reorder scan">
      <div className="scan-head">
        <div>
          <h1>Reorder recommendations</h1>
          <p className="sub">
            The agent scans every SKU on a timer with no question asked. Run it now, or replay it as of an earlier date.
          </p>
        </div>
        <div className="row">
          <label className="field">
            Replay as of (optional)
            <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
          </label>
          <label className="field">
            Max SKUs to size
            <input type="number" min={1} max={500} value={limit} onChange={(e) => setLimit(e.target.value)} style={{ width: 90 }} />
          </label>
          <button className="btn primary" onClick={() => run()} disabled={busy}>
            {busy ? <><span className="spinner" />Scanning…</> : "Run scan now"}
          </button>
        </div>
      </div>
      {busy && <p className="hint">Fitting a forecast for each triggered SKU. The first scan after a restart can take up to a minute.</p>}
      {error && <div className="notice error" role="alert" style={{ marginTop: 12, marginBottom: 0 }}>{error}</div>}

      {stale && last && (
        <div className="notice warn" style={{ marginTop: 12, marginBottom: 0 }}>
          <strong>The stock figures are {last.snapshot_age_days} days old</strong> (dated {day(last.snapshot_date)}), so
          runway checks were skipped for {last.not_assessed.snapshot_stale ?? 0} SKUs rather than guessed. Event
          recommendations are still shown, at low confidence.{" "}
          {last.snapshot_date && (
            <button className="btn sm" onClick={() => { setAsOf(last.snapshot_date!); run(last.snapshot_date!); }} disabled={busy}>
              Replay at {day(last.snapshot_date)}
            </button>
          )}
        </div>
      )}

      {last && (
        <>
          <div className="chips" aria-label="Last scan summary">
            <span className="chip">Last scan <b>{when(last.finished_at)}</b></span>
            <span className="chip">As of <b>{day(last.as_of)}</b>{last.replay ? " (replay)" : ""}</span>
            <span className="chip">SKUs checked <b>{last.skus_in_snapshot}</b></span>
            <span className="chip">Triggered <b>{last.triggered}</b></span>
            <span className="chip">Sized <b>{last.sized}</b></span>
            {last.deferred_count > 0 && <span className="chip">Deferred <b>{last.deferred_count}</b></span>}
            {notAssessed.map(([k, n]) => (
              <span className="chip" key={k}>Not assessed: <b>{n}</b> {NOT_ASSESSED_LABEL[k] ?? k}</span>
            ))}
          </div>
          <details className="more">
            <summary>How this scan went</summary>
            <ul>{last.scan_trace.map((t, i) => <li key={i}>{t}</li>)}</ul>
            {last.deferred_count > 0 && (
              <p>
                {last.deferred_count} triggered SKUs were not sized this time (most urgent are done first). Raise “Max SKUs
                to size” to include them. Examples: {last.deferred.slice(0, 8).join(", ")}…
              </p>
            )}
          </details>
        </>
      )}
    </section>
  );
}
