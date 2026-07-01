import Link from "next/link";
import { DataNotReadyError, loadBacktestReport } from "@/lib/data";
import type { BacktestReportFile } from "@/lib/types";

function loadReport(): BacktestReportFile | null {
  try {
    return loadBacktestReport();
  } catch (err) {
    if (err instanceof DataNotReadyError) return null;
    throw err;
  }
}

function pct(x: number) {
  return `${Math.round(x * 1000) / 10}%`;
}

export default function MethodologyPage() {
  const report = loadReport();

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 pb-16 pt-10">
      <header>
        <div className="flex items-baseline justify-between gap-4">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Methodology &amp; backtest
          </h1>
          <Link
            href="/"
            className="whitespace-nowrap text-xs font-medium text-sky-400 hover:text-sky-300 hover:underline"
          >
            &larr; Back to draft builder
          </Link>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-slate-400">
          CompStrength blends recent patch-weighted pro play with solo queue
          performance, plus champion-pair synergy and lane matchup history,
          into a logistic model of blue-side win probability. The numbers
          below come from holding out games the model never trained on.
        </p>
      </header>

      {!report && (
        <div className="rounded-md border border-amber-800 bg-amber-950/40 px-4 py-3 text-sm text-amber-300">
          The backtest report hasn&apos;t been generated yet. Run the data
          pipeline from the repo root to produce{" "}
          <code className="font-mono">data/backtest_report.json</code>.
        </div>
      )}

      {report && (
        <>
          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
              Headline metrics
            </h2>
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Accuracy</dt>
                <dd className="text-lg font-semibold text-slate-100">
                  {pct(report.metrics.accuracy)}
                </dd>
                <dd className="text-xs text-slate-500">
                  vs. {pct(report.metrics.baselineAccuracy)} baseline
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Log loss</dt>
                <dd className="text-lg font-semibold text-slate-100">
                  {report.metrics.logLoss.toFixed(3)}
                </dd>
                <dd className="text-xs text-slate-500">
                  vs. {report.metrics.coinFlipLogLoss.toFixed(3)} coin-flip
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Brier score</dt>
                <dd className="text-lg font-semibold text-slate-100">
                  {report.metrics.brierScore.toFixed(3)}
                </dd>
                <dd className="text-xs text-slate-500">lower is better</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Folds</dt>
                <dd className="text-lg font-semibold text-slate-100">{report.folds}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Test games</dt>
                <dd className="text-lg font-semibold text-slate-100">{report.testGames}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">Generated</dt>
                <dd className="text-sm text-slate-300">
                  {new Date(report.generatedAt).toLocaleDateString()}
                </dd>
              </div>
            </dl>
            {report.note && (
              <p className="mt-4 text-xs text-slate-500">{report.note}</p>
            )}
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
              Calibration
            </h2>
            <p className="mb-3 text-xs text-slate-500">
              For predictions bucketed by predicted win probability, how
              often did the predicted side actually win?
            </p>
            {report.calibration.length === 0 ? (
              <p className="text-sm text-slate-500">No calibration buckets available.</p>
            ) : (
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-500">
                  <tr>
                    <th className="pb-1 pr-2">Predicted bucket</th>
                    <th className="pb-1 pr-2">Predicted mean</th>
                    <th className="pb-1 pr-2">Actual win rate</th>
                    <th className="pb-1">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {report.calibration.map((b) => (
                    <tr key={b.bucket} className="border-t border-slate-800">
                      <td className="py-1 pr-2 font-mono text-slate-300">{b.bucket}</td>
                      <td className="py-1 pr-2 text-slate-400">{pct(b.predictedMean)}</td>
                      <td className="py-1 pr-2 text-slate-400">{pct(b.actualWinRate)}</td>
                      <td className="py-1 text-slate-500">{b.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </div>
  );
}
