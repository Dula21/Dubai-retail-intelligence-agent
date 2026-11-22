// Types mirror the FastAPI responses (backend/agents/reorder_scan.py, agent_router.py). The browser calls the
// same-origin proxy at /api/agent/*, which forwards to the backend.

export type Status = "pending" | "approved" | "modified" | "rejected";
export type Urgency = "late" | "now" | "upcoming";

export interface Recommendation {
  rec_id: string;
  sku_id: string;
  product_name: string;
  category: string;
  trigger: string;
  event: string | null;
  urgency: Urgency;
  order_qty: number;
  order_by_date: string | null;
  days_to_order_by: number;
  window_start: string | null;
  window_end: string | null;
  forecast_units: number | null;
  safety_multiplier: number | null;
  target_units: number | null;
  on_hand_snapshot: number | null;
  snapshot_date: string | null;
  snapshot_age_days: number | null;
  projected_on_hand: number | null;
  open_orders: number | null;
  confidence: number;
  confidence_basis: string;
  flags: string[];
  headline: string;
  reasoning_trace: string[];
  status: Status;
  as_of: string;
  replay: boolean;
  created_at: string;
  decided_by?: string | null;
  decided_at?: string | null;
  decision_reason?: string | null;
  final_qty?: number | null;
  modification_delta_pct?: number;
}

export interface HistoryEvent {
  event_type: string;
  actor: string;
  reason: string | null;
  qty_before: number | null;
  qty_after: number | null;
  created_at: string;
}

export interface ScanSummary {
  as_of: string;
  replay: boolean;
  snapshot_date: string | null;
  snapshot_age_days: number | null;
  skus_in_snapshot: number;
  triggered: number;
  sized: number;
  deferred_count: number;
  deferred: string[];
  not_assessed: Record<string, number>;
  skipped: Record<string, number>;
  persisted?: boolean;
  scan_trace: string[];
  finished_at: string;
}

export interface ListResponse {
  count: number;
  source: "postgres" | "memory";
  last_scan: ScanSummary | null;
  recommendations: Recommendation[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/agent/${path}`, {
    ...init,
    headers: { "content-type": "application/json" },
    cache: "no-store",
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ")
          : text.slice(0, 200) || res.statusText;
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  list: () => call<ListResponse>("recommendations"),
  get: (id: string) =>
    call<{ source: string; recommendation: Recommendation; history: HistoryEvent[] }>(`recommendations/${id}`),
  scan: (body: { as_of?: string; limit?: number }) => call<ScanSummary & { recommendations: Recommendation[] }>("scan", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  decide: (id: string, body: { decision: "approve" | "reject" | "modify"; actor: string; reason?: string; quantity?: number }) =>
    call<{ persisted: boolean; recommendation: Recommendation }>(`recommendations/${id}/decision`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  ask: (query: string) => call<AskResponse>("query", { method: "POST", body: JSON.stringify({ query }) }),
  health: () => call<{ status: string }>("health"),
};

export interface AskResponse {
  recommendation: string;
  confidence: number;
  reasoning_trace: string[];
  order_recommendations?: {
    sku_id: string;
    order_qty: number | null;
    window_start: string | null;
    window_end: string | null;
    indicative_only?: boolean;
    order_by_date?: string | null;
    coverage_estimate: number | null;
    flags: string[];
  }[];
  seasonality?: { upcoming_event: string | null; days_until_event: number | null; alert_level: string } | null;
  fallback_triggered: boolean;
  query_language: string;
  intent: string;
}
