/** Decimal ("European") odds implied by a probability: the break-even
 *  bookmaker price for that side. 60% -> 1.67, 40% -> 2.50. */
function decimalOdds(probability: number): string {
  return (1 / Math.max(probability, 1e-6)).toFixed(2);
}

export function WinBar({
  blueWinProbability,
}: {
  blueWinProbability: number;
}) {
  const redWinProbability = 1 - blueWinProbability;
  const bluePct = Math.round(blueWinProbability * 1000) / 10;
  const redPct = Math.round(redWinProbability * 1000) / 10;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm font-semibold">
        <span className="text-sky-400">
          Blue {bluePct}%{" "}
          <span className="ml-1 rounded bg-sky-950/60 px-1.5 py-0.5 font-mono text-xs text-sky-300">
            {decimalOdds(blueWinProbability)}
          </span>
        </span>
        <span className="text-rose-400">
          <span className="mr-1 rounded bg-rose-950/60 px-1.5 py-0.5 font-mono text-xs text-rose-300">
            {decimalOdds(redWinProbability)}
          </span>{" "}
          Red {redPct}%
        </span>
      </div>
      <div className="flex h-4 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full bg-sky-500 transition-all"
          style={{ width: `${bluePct}%` }}
        />
        <div
          className="h-full bg-rose-500 transition-all"
          style={{ width: `${redPct}%` }}
        />
      </div>
    </div>
  );
}
