import type { NotablePairs as NotablePairsData } from "@/lib/types";

function pct(x: number) {
  return `${x >= 0 ? "+" : ""}${Math.round(x * 1000) / 10}%`;
}

export function NotablePairs({ notablePairs }: { notablePairs: NotablePairsData }) {
  const { synergy, matchup } = notablePairs;
  if (synergy.length === 0 && matchup.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 border-t border-slate-800 pt-4 text-sm">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Notable pairings
      </h3>
      <ul className="flex flex-col gap-1.5">
        {synergy.map(({ pair, residual }) => (
          <li key={`synergy-${pair[0]}-${pair[1]}`} className="flex items-center gap-2">
            <span className="rounded bg-emerald-950/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
              Synergy
            </span>
            <span className="text-slate-300">
              {pair[0]} + {pair[1]}
            </span>
            <span className={residual >= 0 ? "text-emerald-400" : "text-rose-400"}>
              {pct(residual)} historically together
            </span>
          </li>
        ))}
        {matchup.map(({ pair, residual }) => (
          <li key={`matchup-${pair[0]}-${pair[1]}`} className="flex items-center gap-2">
            <span className="rounded bg-amber-950/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-400">
              Matchup
            </span>
            <span className="text-slate-300">
              {pair[0]} vs {pair[1]}
            </span>
            <span className={residual >= 0 ? "text-emerald-400" : "text-rose-400"}>
              {pct(residual)} lane edge
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
