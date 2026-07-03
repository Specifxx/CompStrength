import localFont from "next/font/local";

/**
 * JetBrains Mono, self-hosted (woff2 bundled under app/fonts) so it needs no
 * build-time or runtime network access and works under a strict CSP. This is
 * the site-wide "market terminal" typeface — one consistent monospace across
 * headings, labels, numbers, and prose — giving RiftCompare the trading
 * -terminal look, with tabular figures that keep columns of odds and Elo
 * aligned.
 */
export const terminalFont = localFont({
  src: [
    { path: "./fonts/JetBrainsMono-Regular.woff2", weight: "400", style: "normal" },
    { path: "./fonts/JetBrainsMono-Medium.woff2", weight: "500", style: "normal" },
    { path: "./fonts/JetBrainsMono-SemiBold.woff2", weight: "600", style: "normal" },
    { path: "./fonts/JetBrainsMono-Bold.woff2", weight: "700", style: "normal" },
  ],
  display: "swap",
  variable: "--font-terminal",
  fallback: [
    "ui-monospace",
    "SFMono-Regular",
    "Menlo",
    "Consolas",
    "Liberation Mono",
    "monospace",
  ],
});
