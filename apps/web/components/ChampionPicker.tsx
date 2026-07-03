"use client";

import { useMemo, useRef, useState } from "react";
import { ChampionIcon } from "./ChampionIcon";
import type { ChampionListItem, Role } from "@/lib/types";

const CONFIDENCE_DOT: Record<string, string> = {
  low: "bg-amber-500",
  medium: "bg-sky-500",
  high: "bg-emerald-500",
};

/** Splits `text` into [before, match, after] around the first case-insensitive occurrence of `query`. */
function splitMatch(text: string, query: string): [string, string, string] | null {
  if (!query) return null;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return null;
  return [text.slice(0, idx), text.slice(idx, idx + query.length), text.slice(idx + query.length)];
}

function HighlightedName({ name, query }: { name: string; query: string }) {
  const parts = splitMatch(name, query.trim());
  if (!parts) return <span>{name}</span>;
  const [before, match, after] = parts;
  return (
    <span>
      {before}
      <span className="font-semibold text-sky-300">{match}</span>
      {after}
    </span>
  );
}

/** Focus the draft input at position ``seq`` (0-9 across blue then red), or
 *  the Predict button when advancing past the last one — the backbone of the
 *  keyboard-only flow (Tab / comma / Enter fills the current pick and jumps
 *  to the next). */
function focusSeq(seq: number) {
  const el = document.querySelector<HTMLElement>(`[data-draft-seq="${seq}"]`);
  if (el) {
    el.focus();
  } else {
    document.querySelector<HTMLElement>("[data-draft-predict]")?.focus();
  }
}

export function ChampionPicker({
  role,
  side,
  value,
  champions,
  takenElsewhere,
  onChange,
  seq,
}: {
  role: Role;
  side: "blue" | "red";
  value: string | null;
  champions: ChampionListItem[];
  takenElsewhere: Set<string>;
  onChange: (champion: string | null) => void;
  /** Position in the 0-9 keyboard-navigation order (blue TOP..SUP, then red). */
  seq?: number;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  function selectChampion(name: string) {
    onChange(name);
    setQuery("");
    setOpen(false);
    setHighlightIndex(0);
  }

  /** Accept the best current match (highlighted suggestion, or the top one if
   *  the user just typed) and jump to the next pick. If nothing matches, keep
   *  whatever's already selected and still advance — so a full keyboard run
   *  never gets stuck. */
  function commitAndAdvance() {
    const chosen = suggestions[highlightIndex] ?? suggestions[0];
    if (open && chosen) selectChampion(chosen.name);
    if (seq !== undefined) focusSeq(seq + 1);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    // Comma is a dedicated "accept + next" key (never typed into the field).
    if (e.key === ",") {
      e.preventDefault();
      commitAndAdvance();
      return;
    }
    // Plain Tab (not Shift+Tab) fills the current pick and moves to the next
    // champion input specifically, skipping the clear button / team fields.
    if (e.key === "Tab" && !e.shiftKey && seq !== undefined) {
      e.preventDefault();
      commitAndAdvance();
      return;
    }
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
      commitAndAdvance();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      setQuery("");
      inputRef.current?.blur();
    }
  }

  return (
    <div className="relative" ref={containerRef}>
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-400">
        {role}
      </label>
      <div className="relative flex items-center">
        {value && !open && (
          <span className="pointer-events-none absolute left-2">
            <ChampionIcon champion={value} size={20} />
          </span>
        )}
        <input
          ref={inputRef}
          type="text"
          data-draft-seq={seq}
          value={open ? query : value ?? ""}
          placeholder="Search champion..."
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
          className={`w-full rounded-md border border-slate-700 bg-slate-900 py-2 pr-3 text-sm text-slate-100 outline-none ${
            value && !open ? "pl-8" : "pl-3"
          } ${accent}`}
        />
      </div>
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
          {suggestions.map((c, i) => (
            <li key={c.name}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  selectChampion(c.name);
                }}
                onMouseEnter={() => setHighlightIndex(i)}
                className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm text-slate-200 ${
                  i === highlightIndex ? "bg-slate-800" : ""
                }`}
              >
                <span className="flex items-center gap-2">
                  <ChampionIcon champion={c.name} size={20} />
                  <HighlightedName name={c.name} query={query} />
                </span>
                <span className="flex items-center gap-2 text-xs text-slate-500">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${CONFIDENCE_DOT[c.sampleConfidence]}`}
                    title={`${c.sampleConfidence} sample confidence`}
                  />
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
                    {c.primaryRole}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
