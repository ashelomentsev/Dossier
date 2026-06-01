// Formatting helpers for extracted labels.

import { Labels } from "./types.ts";

function titleCase(key: string): string {
  const s = key.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Escape Telegram Markdown (v1) special characters in user-derived values. */
function escapeMarkdown(text: string): string {
  return text.replace(/([_*`\[])/g, "\\$1");
}

/** Render labels as a Markdown list: "*Name*: Sarah". */
export function formatLabels(labels: Labels): string {
  const lines: string[] = [];
  for (const [key, value] of Object.entries(labels)) {
    if (value == null) continue;
    const text = Array.isArray(value) ? value.join(", ") : String(value);
    if (!text.trim()) continue;
    lines.push(`*${titleCase(key)}*: ${escapeMarkdown(text)}`);
  }
  return lines.join("\n");
}

/** The person's display name, if the labels carry one. */
export function personName(labels: Labels): string | null {
  const name = labels.name;
  if (typeof name === "string" && name.trim()) return name.trim();
  return null;
}
