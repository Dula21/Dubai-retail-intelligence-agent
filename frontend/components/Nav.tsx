"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function Nav() {
  const path = usePathname();
  const [up, setUp] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () => api.health().then(() => alive && setUp(true)).catch(() => alive && setUp(false));
    check();
    const t = setInterval(check, 30000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  return (
    <header className="topbar">
      <div className="brand">DRI Agent<small>Dubai Retail Intelligence</small></div>
      <nav className="nav" aria-label="Main">
        <Link href="/" aria-current={path === "/" ? "page" : undefined}>Recommendations</Link>
        <Link href="/ask" aria-current={path === "/ask" ? "page" : undefined}>Ask the agent</Link>
      </nav>
      <div className="spacer" />
      <div className="health" role="status">
        <span className={`dot ${up === null ? "" : up ? "up" : "down"}`} />
        {up === null ? "Checking API…" : up ? "API connected" : "API unreachable"}
      </div>
    </header>
  );
}
