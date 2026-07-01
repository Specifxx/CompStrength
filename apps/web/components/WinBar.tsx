export function WinBar({
  blueWinProbability,
}: {
  blueWinProbability: number;
}) {
  const bluePct = Math.round(blueWinProbability * 1000) / 10;
  const redPct = Math.round((1 - blueWinProbability) * 1000) / 10;

  return (
    <div>
      <div className="mb-1 flex justify-between text-sm font-semibold">
        <span className="text-sky-400">Blue {bluePct}%</span>
        <span className="text-rose-400">Red {redPct}%</span>
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
