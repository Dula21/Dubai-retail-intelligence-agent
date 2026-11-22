import { Fragment } from "react";

/** Minimal, safe rendering of the model's text: **bold**, "- " bullets, blank-line paragraphs. No HTML is injected. */
function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : <Fragment key={i}>{part}</Fragment>);
}

export default function Rich({ text }: { text: string }) {
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) {
      blocks.push(<ul key={`u${blocks.length}`}>{bullets.map((b, i) => <li key={i}>{inline(b)}</li>)}</ul>);
      bullets = [];
    }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (/^[-*•]\s+/.test(line)) bullets.push(line.replace(/^[-*•]\s+/, ""));
    else { flush(); if (line) blocks.push(<p key={`p${blocks.length}`}>{inline(line)}</p>); }
  }
  flush();
  return <div className="answer">{blocks}</div>;
}
