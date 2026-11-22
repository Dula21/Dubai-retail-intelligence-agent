import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "DRI Agent — reorder recommendations",
  description: "Autonomous reorder recommendations for Dubai retailers, with reasoning traces and human override.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Nav />
          <div className="notice">
            <strong>Synthetic data.</strong> Every number here comes from a generated Dubai retail dataset, so this
            demonstrates the pipeline, not real demand. Confidence is the measured share of past held-out windows that
            a target of this kind covered, not a probability for one SKU.
          </div>
          {children}
        </div>
      </body>
    </html>
  );
}
