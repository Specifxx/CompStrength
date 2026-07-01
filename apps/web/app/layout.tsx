import type { Metadata } from "next";
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
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100">
        {children}
      </body>
    </html>
  );
}
