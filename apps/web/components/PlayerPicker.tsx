"use client";

import { useMemo, useRef, useState } from "react";
import type { PlayersFile, Role } from "@/lib/types";

/** Compact, optional player autocomplete for one roster seat.
 *
 * Pre-filled from the selected team's roster but fully editable, so a user can
 * swap in a substitute. Suggestions come from the global players index
 * (data/players.json), sorted by Elo. A free-typed name that isn't in the
 * index is still accepted — the model then treats that player as perfectly
 * average (Elo 1500, no champion history), exactly like the pipeline. */
export function PlayerPicker({
  role,
  side,
  value,
  players,
  onChange,
}: {
  role: Role;
  side: "blue" | "red";
  value: string | null;
  players: PlayersFile;
  onChange: (player: string | null) => void;
}) {
  const [query, setQuery] = useState<string | null>(null); // null = show value
  const [open, setOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const allPlayers = useMemo(
    () =>
      Object.entries(players.players)
        .map(([name, p]) => ({ name, elo: p.elo, games: p.games }))
        .sort((a, b) => b.elo - a.elo),
    [players],
  );

  const suggestions = useMemo(() => {
    const q = (query ?? "").trim().toLowerCase();
    return allPlayers
      .filter((p) => (q ? p.name.toLowerCase().includes(q) : true))
      .slice(0, 8);
  }, [allPlayers, query]);

  const accent = side === "blue" ? "focus:border-sky-400" : "focus:border-rose-400";

  function commit(name: string | null) {
    onChange(name && name.trim() ? name.trim() : null);
    setQuery(null);
    setOpen(false);
    setHighlightIndex(0);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      if (suggestions.length) setHighlightIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (suggestions.length)
        setHighlightIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === "Enter" || e.key === "Tab") {
      // Accept the highlighted suggestion if the dropdown is open with a
      // query; otherwise keep the free-typed text. Tab keeps its default
      // focus-advance (players aren't part of the champion seq flow).
      if (open && query !== null && suggestions[highlightIndex]) {
        if (e.key === "Enter") e.preventDefault();
        commit(suggestions[highlightIndex].name);
      } else if (query !== null) {
        commit(query);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setQuery(null);
      setOpen(false);
      inputRef.current?.blur();
    }
  }

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        aria-label={`${role} player`}
        value={query !== null ? query : value ?? ""}
        placeholder={role}
        onFocus={() => {
          setOpen(true);
          setQuery("");
          setHighlightIndex(0);
        }}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setHighlightIndex(0);
        }}
        onKeyDown={handleKeyDown}
        onBlur={() => setTimeout(() => commit(query !== null ? query : value), 120)}
        className={`w-full rounded border border-slate-800 bg-slate-900/70 px-2 py-1 text-xs text-slate-200 outline-none ${accent}`}
      />
      {open && suggestions.length > 0 && (
        <ul className="absolute z-20 mt-1 max-h-56 w-full overflow-auto rounded-md border border-slate-700 bg-slate-900 shadow-lg">
          {suggestions.map((p, i) => (
            <li key={p.name}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  commit(p.name);
                }}
                onMouseEnter={() => setHighlightIndex(i)}
                className={`flex w-full items-center justify-between gap-2 px-2 py-1 text-left text-xs text-slate-200 ${
                  i === highlightIndex ? "bg-slate-800" : ""
                }`}
              >
                <span className="truncate">{p.name}</span>
                <span className="shrink-0 tabular-nums text-slate-500">
                  {Math.round(p.elo)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
