"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ChampionPicker } from "./ChampionPicker";
import { TeamPicker } from "./TeamPicker";
import { ResultBreakdown } from "./ResultBreakdown";
import { NotablePairs } from "./NotablePairs";
import { WinBar } from "./WinBar";
import {
  IncompleteDraftError,
  UnknownChampionError,
  predictMatchup,
} from "@/lib/predict";
import {
  ROLES,
  type BacktestReportFile,
  type ChampionListItem,
  type ChampionRatingsFile,
  type DraftTeam,
  type ModelFile,
  type PredictResponse,
  type SynergyFile,
  type TeamsFile,
} from "@/lib/types";

const EMPTY_TEAM: DraftTeam = {
  TOP: null,
  JUNGLE: null,
  MID: null,
  BOTTOM: null,
  SUPPORT: null,
};

export function DraftBuilder({
  champions,
  ratings,
  patch,
  model,
  synergy,
  backtest,
  teams,
}: {
  champions: ChampionListItem[];
  ratings: ChampionRatingsFile;
  patch: string;
  model: ModelFile;
  synergy: SynergyFile;
  backtest?: BacktestReportFile | null;
  teams?: TeamsFile | null;
}) {
  const [blue, setBlue] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [red, setRed] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [blueOrg, setBlueOrg] = useState<string | null>(null);
  const [redOrg, setRedOrg] = useState<string | null>(null);

  const blueTaken = useMemo(
    () => new Set(Object.values(blue).filter((v): v is string => !!v)),
    [blue],
  );
  const redTaken = useMemo(
    () => new Set(Object.values(red).filter((v): v is string => !!v)),
    [red],
  );
  const allTaken = useMemo(() => new Set([...blueTaken, ...redTaken]), [blueTaken, redTaken]);

  const blueComplete = ROLES.every((r) => blue[r]);
  const redComplete = ROLES.every((r) => red[r]);
  const canPredict = blueComplete && redComplete;

  function handlePredict() {
    setError(null);
    try {
      // Computed entirely client-side — no network round-trip. lib/predict.ts
      // is pure/isomorphic TS, so it runs identically here and on the
      // /api/predict route (which stays available for programmatic use).
      const prediction = predictMatchup(blue, red, ratings, model, synergy, {
        blueTeam: blueOrg,
        redTeam: redOrg,
        teams,
      });
      setResult(prediction);
    } catch (err) {
      if (err instanceof IncompleteDraftError || err instanceof UnknownChampionError) {
        setError(err.message);
      } else {
        setError("Prediction failed");
      }
      setResult(null);
    }
  }

  function reset() {
    setBlue({ ...EMPTY_TEAM });
    setRed({ ...EMPTY_TEAM });
    setBlueOrg(null);
    setRedOrg(null);
    setResult(null);
    setError(null);
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 pb-16 pt-10">
      <header>
        <div className="flex items-baseline justify-between gap-4">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Comp<span className="text-sky-400">Strength</span>
          </h1>
          <Link
            href="/methodology"
            className="whitespace-nowrap text-xs font-medium text-sky-400 hover:text-sky-300 hover:underline"
          >
            Methodology &amp; backtest &rarr;
          </Link>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-slate-400">
          Build a blue-side and red-side draft and estimate win probability
          from recent, patch-weighted pro play blended with solo queue
          performance. Currently calibrated on patch{" "}
          <span className="font-mono text-slate-300">{patch}</span>. This is a
          draft-strength estimate, not a guaranteed outcome — team comp is one
          input among many that decide a pro match.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <section className="rounded-lg border border-sky-900/60 bg-sky-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
            Blue Side
          </h2>
          <div className="flex flex-col gap-3">
            {teams && Object.keys(teams.teams).length > 0 && (
              <TeamPicker side="blue" value={blueOrg} teams={teams} onChange={setBlueOrg} />
            )}
            {ROLES.map((role) => (
              <ChampionPicker
                key={role}
                role={role}
                side="blue"
                value={blue[role]}
                champions={champions}
                takenElsewhere={allTaken}
                onChange={(champion) => setBlue((prev) => ({ ...prev, [role]: champion }))}
              />
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-rose-400">
            Red Side
          </h2>
          <div className="flex flex-col gap-3">
            {teams && Object.keys(teams.teams).length > 0 && (
              <TeamPicker side="red" value={redOrg} teams={teams} onChange={setRedOrg} />
            )}
            {ROLES.map((role) => (
              <ChampionPicker
                key={role}
                role={role}
                side="red"
                value={red[role]}
                champions={champions}
                takenElsewhere={allTaken}
                onChange={(champion) => setRed((prev) => ({ ...prev, [role]: champion }))}
              />
            ))}
          </div>
        </section>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={!canPredict}
          onClick={handlePredict}
          className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          Predict Winner
        </button>
        <button
          type="button"
          onClick={reset}
          className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800"
        >
          Reset
        </button>
        {!canPredict && (
          <span className="text-xs text-slate-500">
            Fill all 10 roles to run a prediction.
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-amber-800 bg-amber-950/40 px-4 py-3 text-sm text-amber-300">
          {error}
        </div>
      )}

      {result && (
        <section className="flex flex-col gap-6 rounded-lg border border-slate-800 bg-slate-900/40 p-5">
          <WinBar blueWinProbability={result.blueWinProbability} />

          {/* The odds chips in the WinBar are the model's FAIR decimal odds
              (1 / probability): the break-even bookmaker price for each side
              if the model probability is right. Offered odds ABOVE the fair
              number = positive expected value per the model (before the
              bookmaker's margin and the model's own error bars). */}
          <p className="text-xs text-slate-500">
            Suggested odds shown are fair decimal odds (1 &divide;
            probability), bookmaker style — if a book offers a side at longer
            odds than the suggested number, the model sees value there.
          </p>

          {result.teamContext ? (
            <div className="flex flex-col gap-1 text-sm text-slate-400">
              <p>
                Team strength applied:{" "}
                <span className="text-sky-300">{result.teamContext.blueTeam}</span>{" "}
                <span className="font-mono text-slate-300">
                  {Math.round(result.teamContext.blueElo)}
                </span>{" "}
                vs{" "}
                <span className="text-rose-300">{result.teamContext.redTeam}</span>{" "}
                <span className="font-mono text-slate-300">
                  {Math.round(result.teamContext.redElo)}
                </span>{" "}
                Elo — team strength is the strongest single predictor; the draft
                adjusts it.
              </p>
              {result.teamContext.blueRoster && result.teamContext.redRoster && (
                <p className="text-xs text-slate-500">
                  Player strength applied (inferred starting fives, avg player
                  Elo{" "}
                  <span className="font-mono text-slate-400">
                    {Math.round(result.teamContext.bluePlayerElo ?? 0)}
                  </span>{" "}
                  vs{" "}
                  <span className="font-mono text-slate-400">
                    {Math.round(result.teamContext.redPlayerElo ?? 0)}
                  </span>
                  , plus each player&apos;s record on the drafted champion):{" "}
                  <span className="text-sky-300/80">
                    {result.teamContext.blueRoster.join(", ")}
                  </span>{" "}
                  vs{" "}
                  <span className="text-rose-300/80">
                    {result.teamContext.redRoster.join(", ")}
                  </span>
                  .
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500">
              {(blueOrg ? 1 : 0) + (redOrg ? 1 : 0) === 1
                ? "Team strength needs BOTH teams selected — only one is set, so this is a draft-only prediction."
                : "Draft-only prediction (no teams selected). Pick both teams above to include team & player strength — the strongest predictors (rosters are inferred automatically)."}
            </p>
          )}

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <ResultBreakdown side="blue" contributions={result.breakdown.blue} />
            <ResultBreakdown side="red" contributions={result.breakdown.red} />
          </div>
          <NotablePairs notablePairs={result.notablePairs} />
          {backtest && Number.isFinite(backtest.metrics.accuracy) ? (
            // Show the HONEST walk-forward held-out numbers (same as the
            // Methodology page), NOT the model.json in-sample metrics, which
            // are inflated by synergy/matchup leakage. The two modes differ a
            // lot (teams carry most of the signal), so show the numbers that
            // match how THIS prediction was made: with-teams metrics when the
            // team term applied, the draft-only companion metrics otherwise.
            // Prefer the current-season slice when present: it matches what a
            // prediction made today actually faces (current-season game, all
            // history available for training).
            <p className="text-xs text-slate-500">
              {result.teamContext || !backtest.draftOnlyMetrics ? (
                <>
                  Held-out accuracy{" "}
                  {(
                    (backtest.currentSeasonMetrics?.accuracy ??
                      backtest.metrics.accuracy) * 100
                  ).toFixed(1)}
                  % vs.{" "}
                  {(
                    (backtest.currentSeasonMetrics?.baselineAccuracy ??
                      backtest.metrics.baselineAccuracy) * 100
                  ).toFixed(1)}
                  % pick-majority baseline; log-loss{" "}
                  {(
                    backtest.currentSeasonMetrics?.logLoss ??
                    backtest.metrics.logLoss
                  ).toFixed(3)}{" "}
                  vs. {backtest.metrics.coinFlipLogLoss.toFixed(3)} coin-flip
                  (current-season walk-forward, teams known).
                </>
              ) : (
                <>
                  Held-out DRAFT-ONLY accuracy{" "}
                  {(
                    (backtest.currentSeasonMetrics?.draftOnlyAccuracy ??
                      backtest.draftOnlyMetrics.accuracy) * 100
                  ).toFixed(1)}
                  % vs.{" "}
                  {(
                    (backtest.currentSeasonMetrics?.baselineAccuracy ??
                      backtest.metrics.baselineAccuracy) * 100
                  ).toFixed(1)}
                  % pick-majority baseline; log-loss{" "}
                  {(
                    backtest.currentSeasonMetrics?.draftOnlyLogLoss ??
                    backtest.draftOnlyMetrics.logLoss
                  ).toFixed(3)}{" "}
                  vs. {backtest.metrics.coinFlipLogLoss.toFixed(3)} coin-flip.
                  Select both teams for the model that reaches{" "}
                  {(
                    (backtest.currentSeasonMetrics?.accuracy ??
                      backtest.metrics.accuracy) * 100
                  ).toFixed(1)}
                  % held-out accuracy.
                </>
              )}{" "}
              See{" "}
              <Link href="/methodology" className="text-sky-400 hover:underline">
                Methodology
              </Link>
              .
            </p>
          ) : (
            <p className="text-xs text-slate-500">
              In-sample (training-set) log-loss{" "}
              {result.modelMetrics.logLoss.toFixed(3)}, accuracy{" "}
              {(result.modelMetrics.accuracy * 100).toFixed(1)}% vs.{" "}
              {(result.modelMetrics.baselineAccuracy * 100).toFixed(1)}% baseline —
              these are optimistic; see the Methodology page for honest held-out
              numbers.
              {result.modelMetrics.note ? ` ${result.modelMetrics.note}` : ""}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
