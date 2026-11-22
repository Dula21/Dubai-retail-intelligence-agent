/**
 * Same-origin proxy to the FastAPI backend.
 *
 * DECISION: the browser talks to this route, and this route talks to the API (server to server).
 *   Alternative: browser calls the API directly, which needs CORS configured on the API.
 *   Why: no backend change, no CORS surface, the backend URL never reaches the browser, and a scan that fits a few
 *   dozen Prophet models (up to a minute on a cold start) is not cut short by a dev-server proxy timeout.
 *   Trade-off: one extra hop (a few ms on localhost). Only an allow-list of paths is forwarded.
 */
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

const BACKEND = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
const ALLOWED = [/^recommendations(\/[\w-]+(\/decision)?)?$/, /^scan$/, /^query$/, /^health$/];

async function forward(req: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path.join("/");
  if (!ALLOWED.some((re) => re.test(path))) {
    return NextResponse.json({ detail: "not found" }, { status: 404 });
  }
  const url = `${BACKEND}/api/v1/agent/${path}${req.nextUrl.search}`;
  try {
    const hasBody = req.method !== "GET" && req.method !== "HEAD";
    const res = await fetch(url, {
      method: req.method,
      headers: { "content-type": "application/json" },
      body: hasBody ? await req.text() : undefined,
      cache: "no-store",
    });
    return new NextResponse(await res.text(), {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") || "application/json" },
    });
  } catch {
    return NextResponse.json(
      { detail: `Cannot reach the API at ${BACKEND}. Is the backend running?` },
      { status: 502 },
    );
  }
}

export { forward as GET, forward as POST };
