import { ChampionIcon } from "./ChampionIcon";
import type { ChampionContribution } from "@/lib/types";

function pct(x: number) {
  return `${Math.round(x * 1000) / 10}%`;
}

export function ResultBreakdown({
  side,
  contributions,
}: {
  side: "blue" | "red";
  contributions: ChampionContribution[];
}) {
  const accent = side === "blue" ? "text-sky-400" : "text-rose-400";

  return (
    <table className="w-full text-left text-sm">
      <caption className={`mb-2 text-left text-xs font-semibold uppercase tracking-wide ${accent}`}>
        {side} side draft strength
      </caption>
      <thead className="text-xs uppercase text-slate-500">
        <tr>
          <th className="pb-1 pr-2">Champion</th>
          <th className="pb-1 pr-2">Role</th>
          <th className="pb-1 pr-2">Blended WR</th>
          <th className="pb-1 pr-2">Pro WR</th>
          <th className="pb-1 pr-2">Solo WR</th>
          <th className="pb-1">Confidence</th>
        </tr>
      </thead>
      <tbody>
        {contributions.map((c) => (
          <tr key={c.champion} className="border-t border-slate-800">
            <td className="py-1 pr-2 font-medium">
              <span className="flex items-center gap-2">
                <ChampionIcon champion={c.champion} size={18} />
                {c.champion}
              </span>
            </td>
            <td className="py-1 pr-2 text-slate-400">{c.role}</td>
            <td className="py-1 pr-2">{pct(c.blendedWinRate)}</td>
            <td className="py-1 pr-2 text-slate-400">
              {pct(c.proWinRate)}{" "}
              <span className="text-slate-600">({c.proGames}g)</span>
            </td>
            <td className="py-1 pr-2 text-slate-400">{pct(c.soloWinRate)}</td>
            <td className="py-1 capitalize text-slate-400">{c.sampleConfidence}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
