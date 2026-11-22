"use client";
import { useCallback, useEffect, useState } from "react";
import { api, HistoryEvent, ListResponse, Recommendation } from "@/lib/api";
import ScanPanel from "@/components/ScanPanel";
import RecList from "@/components/RecList";
import RecDetail from "@/components/RecDetail";

export default function Page() {
  const [data, setData] = useState<ListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<{ rec: Recommendation; history: HistoryEvent[] } | null>(null);

  const refresh = useCallback(async () => {
    try { setData(await api.list()); setError(null); } catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const loadDetail = useCallback(async (id: string) => {
    try { const r = await api.get(id); setDetail({ rec: r.recommendation, history: r.history }); }
    catch (e) { setError((e as Error).message); }
  }, []);

  useEffect(() => { if (selectedId) loadDetail(selectedId); else setDetail(null); }, [selectedId, loadDetail]);

  // On a narrow screen the detail panel sits below the list: bring it into view when an item is chosen.
  const select = (id: string) => {
    setSelectedId(id);
    if (window.matchMedia("(max-width: 980px)").matches) {
      setTimeout(() => document.getElementById("detail")?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    }
  };

  const decided = async () => { await refresh(); if (selectedId) await loadDetail(selectedId); };

  return (
    <>
      {error && <div className="notice error" role="alert">{error}</div>}
      {data?.source === "memory" && (
        <div className="notice warn">
          <strong>The audit database is not connected.</strong> Recommendations are held in memory only: they disappear on
          restart and decisions are not recorded. Run <code>alembic upgrade head</code> and check <code>DATABASE_URL</code>.
        </div>
      )}
      <ScanPanel last={data?.last_scan ?? null} onDone={refresh} />
      <div className="layout">
        <RecList recs={data?.recommendations ?? []} selectedId={selectedId} onSelect={select} />
        <div id="detail">
          {detail && detail.rec.rec_id === selectedId ? (
            <RecDetail rec={detail.rec} history={detail.history} onDecided={decided} />
          ) : (
            <div className="card empty">Select a recommendation to see why the agent raised it and to approve, change or reject it.</div>
          )}
        </div>
      </div>
    </>
  );
}
