"use client";

import { useState } from "react";
import { ddragonChampionImageUrl, useDDragonVersion } from "@/lib/ddragon";

export function ChampionIcon({
  champion,
  size = 24,
  className = "",
}: {
  champion: string;
  size?: number;
  className?: string;
}) {
  const version = useDDragonVersion();
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <span
        className={`inline-block shrink-0 rounded-full bg-slate-700 ${className}`}
        style={{ width: size, height: size }}
        aria-hidden
      />
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element -- external, dynamic Data Dragon CDN URL; not worth next/image config for a hobby project
    <img
      src={ddragonChampionImageUrl(version, champion)}
      alt=""
      width={size}
      height={size}
      className={`shrink-0 rounded-full bg-slate-800 object-cover ${className}`}
      onError={() => setFailed(true)}
    />
  );
}
