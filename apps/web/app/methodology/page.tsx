import Link from "next/link";
import { DataNotReadyError, loadBacktestReport } from "@/lib/data";
import type { BacktestReportFile, BacktestSegment } from "@/lib/types";

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

/** Held-out accuracy table for a set of segments (patches or leagues). The
 *  "edge" column shows accuracy minus that segment's own pick-majority
 *  baseline -- the honest measure of whether the model adds anything there. */
function SegmentTable({
  rows,
  label,
}: {
  rows: BacktestSegment[];
  label: string;
}) {
  if (!rows || rows.length === 0) {
    return <p className="text-sm text-slate-500">No {label} breakdown available.</p>;
  }
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase text-slate-500">
        <tr>
          <th className="pb-1 pr-2">{label}</th>
          <th className="pb-1 pr-2">Games</th>
          <th className="pb-1 pr-2">Accuracy</th>
          <th className="pb-1 pr-2">Baseline</th>
          <th className="pb-1 pr-2">Edge</th>
          <th className="pb-1">Log loss</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const edge = r.accuracy - r.baselineAccuracy;
          return (
            <tr key={r.name} className="border-t border-slate-800">
              <td className="py-1 pr-2 font-mono text-slate-300">{r.name}</td>
              <td className="py-1 pr-2 text-slate-500">{r.testGames.toLocaleString()}</td>
              <td className="py-1 pr-2 text-slate-300">{pct(r.accuracy)}</td>
              <td className="py-1 pr-2 text-slate-500">{pct(r.baselineAccuracy)}</td>
              <td
                className={
                  "py-1 pr-2 " + (edge > 0.0005 ? "text-emerald-400" : edge < -0.0005 ? "text-rose-400" : "text-slate-500")
                }
              >
                {edge >= 0 ? "+" : ""}
                {(edge * 100).toFixed(1)}pp
              </td>
              <td className="py-1 text-slate-500">
                {Number.isFinite(r.logLoss) ? r.logLoss.toFixed(3) : "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
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
            {report.currentSeasonMetrics && (
              <div className="mt-4 rounded-md border border-sky-900/60 bg-sky-950/20 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-sky-400">
                  Current season ({report.currentSeasonMetrics.patchMajor}.x
                  patches, {report.currentSeasonMetrics.testGames.toLocaleString()}{" "}
                  held-out games)
                </p>
                <p className="text-sm text-slate-300">
                  With teams:{" "}
                  <span className="font-semibold text-emerald-400">
                    {pct(report.currentSeasonMetrics.accuracy)}
                  </span>{" "}
                  (log-loss {report.currentSeasonMetrics.logLoss.toFixed(3)}) &middot;
                  Draft only:{" "}
                  <span className="font-semibold text-sky-300">
                    {pct(report.currentSeasonMetrics.draftOnlyAccuracy)}
                  </span>{" "}
                  (log-loss {report.currentSeasonMetrics.draftOnlyLogLoss.toFixed(3)})
                  &middot; baseline {pct(report.currentSeasonMetrics.baselineAccuracy)}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  This slice matches what predictions face today: current-season
                  games, with all prior history available for training. The
                  headline metrics above average over the full multi-season
                  walk-forward (including early folds where the model had
                  little history), so they run lower.
                </p>
              </div>
            )}
            {report.teamFeatureUsed && report.draftOnlyMetrics && (
              <div className="mt-4 rounded-md border border-slate-800 bg-slate-950/40 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  With teams vs. draft only (same held-out games)
                </p>
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase text-slate-500">
                    <tr>
                      <th className="pb-1 pr-2">Inputs</th>
                      <th className="pb-1 pr-2">Accuracy</th>
                      <th className="pb-1 pr-2">Log loss</th>
                      <th className="pb-1">Brier</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-slate-800">
                      <td className="py-1 pr-2 text-slate-300">Teams + draft</td>
                      <td className="py-1 pr-2 text-emerald-400">{pct(report.metrics.accuracy)}</td>
                      <td className="py-1 pr-2 text-slate-300">{report.metrics.logLoss.toFixed(3)}</td>
                      <td className="py-1 text-slate-300">{report.metrics.brierScore.toFixed(3)}</td>
                    </tr>
                    <tr className="border-t border-slate-800">
                      <td className="py-1 pr-2 text-slate-300">Draft only</td>
                      <td className="py-1 pr-2 text-slate-400">{pct(report.draftOnlyMetrics.accuracy)}</td>
                      <td className="py-1 pr-2 text-slate-400">{report.draftOnlyMetrics.logLoss.toFixed(3)}</td>
                      <td className="py-1 text-slate-400">{report.draftOnlyMetrics.brierScore.toFixed(3)}</td>
                    </tr>
                  </tbody>
                </table>
                <p className="mt-2 text-xs text-slate-500">
                  Team strength (Elo over game history) carries most of the
                  predictable signal in pro play; the draft refines it. Select
                  both teams on the draft builder to get the top row&apos;s
                  model.
                </p>
              </div>
            )}
            {report.note && (
              <p className="mt-4 text-xs text-slate-500">{report.note}</p>
            )}
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
              Comparing against bookmaker odds
            </h2>
            <div className="flex flex-col gap-2 text-sm text-slate-400">
              <p>
                Every prediction shows <span className="text-slate-200">fair decimal odds</span>{" "}
                (1 &divide; probability) for each side. If a bookmaker offers
                longer odds than the fair number on a side, the model sees
                positive expected value on that side &mdash; before costs.
              </p>
              <p>
                Be honest about the bar: a bookmaker&apos;s implied
                probabilities contain a built-in margin (typically ~4&ndash;7%
                across both sides), and closing lines on major leagues are
                sharp. To profit you need the model&apos;s calibration edge to
                exceed that margin consistently &mdash; check the log-loss and
                calibration table above, not just accuracy, and treat
                small-sample leagues (see the breakdown below) with extra
                skepticism. One structural caveat: team Elo is anchored across
                leagues only by the few inter-league games (MSI, Worlds, EWC),
                so ratings are most trustworthy for matchups WITHIN a league or
                at international events &mdash; an isolated league&apos;s
                ratings can drift high or low as a block. Nothing on this page
                is betting advice; it&apos;s a measured, walk-forward-validated
                probability estimate with known error bars.
              </p>
            </div>
          </section>

          {report.dataComposition && report.dataComposition.totalGames > 0 && (
            <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
                Data the model is built on
              </h2>
              <p className="mb-4 text-xs text-slate-500">
                {report.dataComposition.totalGames.toLocaleString()} real
                professional games, drawn from the leagues and patches below.
                More recent patches are weighted exponentially more (the
                &ldquo;weight&rdquo; column is each patch&apos;s share of full
                weight); older games still contribute, just less. On top of
                that, premier leagues (LCK + LPL) are up-weighted to carry
                ~70% of the total training weight, and international events
                (MSI/Worlds/EWC) get their own boost &mdash; so the champion
                statistics reflect the highest level of play even though
                minor leagues supply more raw games.
              </p>
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    By patch
                  </h3>
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase text-slate-500">
                      <tr>
                        <th className="pb-1 pr-2">Patch</th>
                        <th className="pb-1 pr-2">Games</th>
                        <th className="pb-1">Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.dataComposition.byPatch.map((p) => (
                        <tr key={p.name} className="border-t border-slate-800">
                          <td className="py-1 pr-2 font-mono text-slate-300">{p.name}</td>
                          <td className="py-1 pr-2 text-slate-400">
                            {p.games.toLocaleString()}
                          </td>
                          <td className="py-1 text-slate-500">
                            {(p.recencyWeight * 100).toFixed(0)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    By league (top 12)
                  </h3>
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase text-slate-500">
                      <tr>
                        <th className="pb-1 pr-2">League</th>
                        <th className="pb-1">Games</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.dataComposition.byLeague.slice(0, 12).map((lg) => (
                        <tr key={lg.name} className="border-t border-slate-800">
                          <td className="py-1 pr-2 font-mono text-slate-300">{lg.name}</td>
                          <td className="py-1 text-slate-400">{lg.games.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {report.dataComposition.byLeague.length > 12 && (
                    <p className="mt-2 text-xs text-slate-600">
                      + {report.dataComposition.byLeague.length - 12} more leagues
                    </p>
                  )}
                </div>
              </div>
            </section>
          )}

          {report.breakdowns &&
            (report.breakdowns.byPatch.length > 0 ||
              report.breakdowns.byLeague.length > 0) && (
              <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
                <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
                  Held-out accuracy by segment
                </h2>
                <p className="mb-4 text-xs text-slate-500">
                  The same walk-forward held-out predictions, split by patch and
                  by league. &ldquo;Edge&rdquo; is accuracy minus that
                  segment&apos;s own pick-majority baseline &mdash; a positive
                  edge means the model beat simply guessing the more common
                  outcome there. Per-segment numbers are noisier the fewer games
                  the segment has.
                </p>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  By patch
                </h3>
                <SegmentTable rows={report.breakdowns.byPatch} label="Patch" />
                <h3 className="mb-2 mt-6 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  By league
                </h3>
                <SegmentTable rows={report.breakdowns.byLeague} label="League" />
              </section>
            )}

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
