"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

// Reasonably-recent fallback in case the versions.json fetch fails, times
// out, or is unreachable (e.g. this dev sandbox has no outbound network
// access to Data Dragon — production browsers will normally succeed).
const FALLBACK_VERSION = "15.1.1";

let cachedVersion: string | null = null;
let inFlight: Promise<string> | null = null;

async function fetchLatestVersion(): Promise<string> {
  if (cachedVersion) return cachedVersion;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    try {
      const res = await fetch("https://ddragon.leagueoflegends.com/api/versions.json", {
        signal: AbortSignal.timeout(2500),
      });
      if (!res.ok) throw new Error(`versions.json responded ${res.status}`);
      const versions = (await res.json()) as unknown;
      if (Array.isArray(versions) && typeof versions[0] === "string") {
        cachedVersion = versions[0];
        return cachedVersion;
      }
      throw new Error("Unexpected versions.json shape");
    } catch {
      cachedVersion = FALLBACK_VERSION;
      return cachedVersion;
    }
  })();

  return inFlight;
}

const DDragonVersionContext = createContext<string>(FALLBACK_VERSION);

export function DDragonVersionProvider({ children }: { children: ReactNode }) {
  const [version, setVersion] = useState<string>(cachedVersion ?? FALLBACK_VERSION);

  useEffect(() => {
    let cancelled = false;
    fetchLatestVersion().then((v) => {
      if (!cancelled) setVersion(v);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <DDragonVersionContext.Provider value={version}>
      {children}
    </DDragonVersionContext.Provider>
  );
}

export function useDDragonVersion(): string {
  return useContext(DDragonVersionContext);
}

// Handful of known irregular Data Dragon champion image id exceptions.
// Keyed lowercase so lookups are resilient to casing differences in the
// upstream data source (e.g. "Jarvan IV" vs "Jarvan Iv").
const IRREGULAR_IDS: Record<string, string> = {
  wukong: "MonkeyKing",
  "nunu & willump": "Nunu",
  "renata glasc": "Renata",
  "jarvan iv": "JarvanIV",
  leblanc: "Leblanc",
};

/**
 * Best-effort slugify of a champion display name into its Data Dragon
 * champion image id. Covers common punctuation-stripping (apostrophes,
 * periods, spaces) plus a handful of hardcoded irregular exceptions.
 * Not guaranteed to be perfect for every champion — a 404 on the resulting
 * image URL is an acceptable graceful degradation for this hobby project.
 */
export function ddragonChampionId(name: string): string {
  const irregular = IRREGULAR_IDS[name.toLowerCase()];
  if (irregular) return irregular;
  return name.replace(/['.]/g, "").replace(/\s+/g, "");
}

export function ddragonChampionImageUrl(version: string, name: string): string {
  return `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/${ddragonChampionId(
    name,
  )}.png`;
}
