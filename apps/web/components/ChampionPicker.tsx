"use client";

import { useMemo, useRef, useState } from "react";
import type { ChampionListItem, Role } from "@/lib/types";

const CONFIDENCE_DOT: Record<string, string> = {
  low: "bg-amber-500",
  medium: "bg-sky-500",
  high: "bg-emerald-500",
};

export function ChampionPicker({
  role,
  side,
  value,
  champions,
  takenElsewhere,
  onChange,
}: {
  role: Role;
  side: "blue" | "red";
  value: string | null;
  champions: ChampionListItem[];
  takenElsewhere: Set<string>;
  onChange: (champion: string | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase();
    return champions
      .filter((c) => !takenElsewhere.has(c.name) || c.name === value)
      .filter((c) => (q ? c.name.toLowerCase().includes(q) : true))
      .sort((a, b) => {
        const aRole = a.primaryRole === role ? 0 : 1;
        const bRole = b.primaryRole === role ? 0 : 1;
        if (aRole !== bRole) return aRole - bRole;
        return b.pickRate - a.pickRate;
      })
      .slice(0, 8);
  }, [champions, query, role, takenElsewhere, value]);

  const accent = side === "blue" ? "focus:border-sky-400" : "focus:border-rose-400";

  return (
    <div className="relative" ref={containerRef}>
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-400">
        {role}
      </label>
      <input
        type="text"
        value={open ? query : value ?? ""}
        placeholder="Search champion..."
        onFocus={() => {
          setOpen(true);
          setQuery("");
        }}
        onChange={(e) => setQuery(e.target.value)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        className={`w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none ${accent}`}
      />
      {value && !open && (
        <button
          type="button"
          aria-label={`Clear ${role}`}
          onClick={() => onChange(null)}
          className="absolute right-2 top-8 text-slate-500 hover:text-slate-300"
        >
          ×
        </button>
      )}
      {open && (
        <ul className="absolute z-10 mt-1 max-h-64 w-full overflow-auto rounded-md border border-slate-700 bg-slate-900 shadow-lg">
          {suggestions.length === 0 && (
            <li className="px-3 py-2 text-sm text-slate-500">No matches</li>
          )}
          {suggestions.map((c) => (
            <li key={c.name}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(c.name);
                  setQuery("");
                  setOpen(false);
                }}
                className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm text-slate-200 hover:bg-slate-800"
              >
                <span>{c.name}</span>
                <span className="flex items-center gap-2 text-xs text-slate-500">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${CONFIDENCE_DOT[c.sampleConfidence]}`}
                    title={`${c.sampleConfidence} sample confidence`}
                  />
                  {c.primaryRole}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
