import type { Metadata } from "next";
import { DDragonVersionProvider } from "@/lib/ddragon";
import { terminalFont } from "./fonts";
import "./globals.css";

export const metadata: Metadata = {
  title: "CompStrength — LoL Pro Draft Win Predictor",
  description:
    "Predict pro League of Legends draft win probability from recent patch-weighted pro play and solo queue champion performance.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`h-full antialiased ${terminalFont.variable}`}>
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100">
        <DDragonVersionProvider>{children}</DDragonVersionProvider>
      </body>
    </html>
  );
}
