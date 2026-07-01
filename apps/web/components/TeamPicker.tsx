"use client";

import { useMemo, useRef, useState } from "react";
import type { TeamsFile } from "@/lib/types";

/** Optional team (org) autocomplete for one side of the draft.
 *
 * Mirrors ChampionPicker's interaction model (type-ahead, arrow keys, click)
 * but is explicitly optional: leaving it blank keeps the prediction
 * draft-only. Suggestions are sorted by Elo (strongest first) and show each
 * team's league, rating, and sample size so a rating built on 3 games is
 * visibly less trustworthy than one built on 40.
 */
export function TeamPicker({
  side,
  value,
  teams,
  onChange,
}: {
  side: "blue" | "red";
  value: string | null;
  teams: TeamsFile;
  onChange: (team: string | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const allTeams = useMemo(
    () =>
      Object.entries(teams.teams).map(([name, t]) => ({ name, ...t })),
    [teams],
  );

  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase();
    return allTeams
      .filter((t) =>
        q ? t.name.toLowerCase().includes(q) || t.league.toLowerCase().includes(q) : true,
      )
      .sort((a, b) => b.elo - a.elo)
      .slice(0, 8);
  }, [allTeams, query]);

  const accent = side === "blue" ? "focus:border-sky-400" : "focus:border-rose-400";

  function selectTeam(name: string) {
    onChange(name);
    setQuery("");
    setOpen(false);
    setHighlightIndex(0);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        setOpen(true);
        setQuery("");
        setHighlightIndex(0);
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (suggestions.length === 0) return;
      setHighlightIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (suggestions.length === 0) return;
      setHighlightIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const chosen = suggestions[highlightIndex];
      if (chosen) selectTeam(chosen.name);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      setQuery("");
      inputRef.current?.blur();
    }
  }

  return (
    <div className="relative">
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-400">
        Team <span className="normal-case text-slate-600">(optional)</span>
      </label>
      <input
        ref={inputRef}
        type="text"
        value={open ? query : value ?? ""}
        placeholder="Any team..."
        onFocus={() => {
          setOpen(true);
          setQuery("");
          setHighlightIndex(0);
        }}
        onChange={(e) => {
          setQuery(e.target.value);
          setHighlightIndex(0);
        }}
        onKeyDown={handleKeyDown}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        className={`w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none ${accent}`}
      />
      {value && !open && (
        <button
          type="button"
          aria-label="Clear team"
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
          {suggestions.map((t, i) => (
            <li key={t.name}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  selectTeam(t.name);
                }}
                onMouseEnter={() => setHighlightIndex(i)}
                className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm text-slate-200 ${
                  i === highlightIndex ? "bg-slate-800" : ""
                }`}
              >
                <span className="truncate">{t.name}</span>
                <span className="flex shrink-0 items-center gap-2 text-xs text-slate-500">
                  {t.league && (
                    <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
                      {t.league}
                    </span>
                  )}
                  <span className="tabular-nums">{Math.round(t.elo)}</span>
                  <span className="text-slate-600">({t.games}g)</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
