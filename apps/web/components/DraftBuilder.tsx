"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ChampionPicker } from "./ChampionPicker";
import { TeamPicker } from "./TeamPicker";
import { PlayerPicker } from "./PlayerPicker";
import { ResultBreakdown } from "./ResultBreakdown";
import { NotablePairs } from "./NotablePairs";
import { WinBar } from "./WinBar";
import {
  IncompleteDraftError,
  UnknownChampionError,
  predictMatchup,
  suggestPicks,
  type PickSuggestion,
} from "@/lib/predict";
import {
  ROLES,
  type BacktestReportFile,
  type ChampionListItem,
  type ChampionRatingsFile,
  type DraftTeam,
  type ModelFile,
  type PlayersFile,
  type PredictResponse,
  type Role,
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

/** Draft role -> Oracle's Elixir roster position code. */
const POSITION_BY_ROLE: Record<Role, string> = {
  TOP: "top",
  JUNGLE: "jng",
  MID: "mid",
  BOTTOM: "bot",
  SUPPORT: "sup",
};

const EMPTY_PLAYERS: (string | null)[] = [null, null, null, null, null];

/** The five roster player names in ROLES order for a selected team (nulls
 *  when the team/roster is unknown), used to pre-fill the editable seats. */
function rosterFor(org: string | null, teams?: TeamsFile | null): (string | null)[] {
  const roster = org ? teams?.teams[org]?.roster : null;
  if (!roster) return [...EMPTY_PLAYERS];
  return ROLES.map(
    (role) => roster.find((s) => s.position === POSITION_BY_ROLE[role])?.player ?? null,
  );
}

/** Clickable best-fit suggestion shown under an empty champion slot: the
 *  available champion that most raises this side's win probability given the
 *  current draft, and that resulting win %. Clicking it fills the slot. */
function SuggestionLine({
  suggestion,
  side,
  onPick,
}: {
  suggestion: PickSuggestion | undefined;
  side: "blue" | "red";
  onPick: (champion: string) => void;
}) {
  if (!suggestion) return null;
  const accent = side === "blue" ? "text-sky-300" : "text-rose-300";
  return (
    <button
      type="button"
      onClick={() => onPick(suggestion.champion)}
      title={`Fill with ${suggestion.champion}`}
      className="mt-1 self-start text-[11px] text-slate-500 transition hover:text-slate-300"
    >
      ↳ best fit <span className={`font-medium ${accent}`}>{suggestion.champion}</span>{" "}
      <span className="tabular-nums text-slate-400">
        {(suggestion.sideWinProbability * 100).toFixed(1)}%
      </span>
    </button>
  );
}

/** Editable roster block shown under a side's team picker once a team is
 *  selected: five player seats pre-filled from the roster, each swap-able.
 *  Module-level (not nested in DraftBuilder) so it keeps a stable identity
 *  across renders — otherwise every keystroke would remount the inputs and
 *  drop focus. */
function RosterEditor({
  side,
  seats,
  players,
  onSeat,
}: {
  side: "blue" | "red";
  seats: (string | null)[];
  players: PlayersFile;
  onSeat: (i: number, name: string | null) => void;
}) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950/40 p-2">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
        Players (auto-filled from roster — edit to swap a sub)
      </p>
      <div className="grid grid-cols-1 gap-1.5">
        {ROLES.map((role, i) => (
          <div key={role} className="flex items-center gap-2">
            <span className="w-8 shrink-0 text-[10px] font-medium uppercase text-slate-500">
              {role.slice(0, 3)}
            </span>
            <PlayerPicker
              role={role}
              side={side}
              value={seats[i]}
              players={players}
              onChange={(name) => onSeat(i, name)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export function DraftBuilder({
  champions,
  ratings,
  patch,
  model,
  synergy,
  backtest,
  teams,
  players,
}: {
  champions: ChampionListItem[];
  ratings: ChampionRatingsFile;
  patch: string;
  model: ModelFile;
  synergy: SynergyFile;
  backtest?: BacktestReportFile | null;
  teams?: TeamsFile | null;
  players?: PlayersFile | null;
}) {
  const [blue, setBlue] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [red, setRed] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [blueOrg, setBlueOrg] = useState<string | null>(null);
  const [redOrg, setRedOrg] = useState<string | null>(null);
  // Editable per-seat player names in ROLES order; pre-filled from the
  // selected team's roster, overridable to swap in a substitute.
  const [bluePlayers, setBluePlayers] = useState<(string | null)[]>([...EMPTY_PLAYERS]);
  const [redPlayers, setRedPlayers] = useState<(string | null)[]>([...EMPTY_PLAYERS]);

  // Selecting/clearing a team pre-fills (or clears) its five editable seats.
  function pickBlueOrg(org: string | null) {
    setBlueOrg(org);
    setBluePlayers(rosterFor(org, teams));
  }
  function pickRedOrg(org: string | null) {
    setRedOrg(org);
    setRedPlayers(rosterFor(org, teams));
  }
  function setBlueSeat(i: number, name: string | null) {
    setBluePlayers((prev) => prev.map((p, j) => (j === i ? name : p)));
  }
  function setRedSeat(i: number, name: string | null) {
    setRedPlayers((prev) => prev.map((p, j) => (j === i ? name : p)));
  }
  const showPlayers = !!players && Object.keys(players.players).length > 0;

  const blueTaken = useMemo(
    () => new Set(Object.values(blue).filter((v): v is string => !!v)),
    [blue],
  );
  const redTaken = useMemo(
    () => new Set(Object.values(red).filter((v): v is string => !!v)),
    [red],
  );
  const allTaken = useMemo(() => new Set([...blueTaken, ...redTaken]), [blueTaken, redTaken]);

  // Dynamic best-fit suggestion for each EMPTY slot: the available champion
  // that most raises that side's win probability given the current partial
  // draft + optional teams. Recomputed only when a pick / team / player
  // changes (not per keystroke).
  const suggestions = useMemo(
    () =>
      suggestPicks(blue, red, ratings, model, synergy, champions, {
        blueTeam: blueOrg,
        redTeam: redOrg,
        teams,
        players,
        bluePlayers,
        redPlayers,
      }),
    [blue, red, ratings, model, synergy, champions, blueOrg, redOrg, teams, players, bluePlayers, redPlayers],
  );

  const blueComplete = ROLES.every((r) => blue[r]);
  const redComplete = ROLES.every((r) => red[r]);
  const canPredict = blueComplete && redComplete;

  function runPredict(
    b: DraftTeam,
    r: DraftTeam,
    bOrg: string | null,
    rOrg: string | null,
    bPlayers: (string | null)[],
    rPlayers: (string | null)[],
  ) {
    setError(null);
    try {
      // Computed entirely client-side — no network round-trip. lib/predict.ts
      // is pure/isomorphic TS, so it runs identically here and on the
      // /api/predict route (which stays available for programmatic use).
      const prediction = predictMatchup(b, r, ratings, model, synergy, {
        blueTeam: bOrg,
        redTeam: rOrg,
        teams,
        players,
        bluePlayers: bPlayers,
        redPlayers: rPlayers,
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

  function handlePredict() {
    runPredict(blue, red, blueOrg, redOrg, bluePlayers, redPlayers);
  }

  /** Swap the entire blue and red setups — drafts, teams, and player seats.
   *  Re-runs the prediction in place if one is already showing, so you can see
   *  the mirror instantly (blue-side bias flips too). */
  function swapSides() {
    setBlue(red);
    setRed(blue);
    setBlueOrg(redOrg);
    setRedOrg(blueOrg);
    setBluePlayers(redPlayers);
    setRedPlayers(bluePlayers);
    if (result) {
      runPredict(red, blue, redOrg, blueOrg, redPlayers, bluePlayers);
    }
  }

  function reset() {
    setBlue({ ...EMPTY_TEAM });
    setRed({ ...EMPTY_TEAM });
    setBlueOrg(null);
    setRedOrg(null);
    setBluePlayers([...EMPTY_PLAYERS]);
    setRedPlayers([...EMPTY_PLAYERS]);
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
        <p className="mt-2 text-xs text-slate-500">
          Keyboard: type a champion and press{" "}
          <kbd className="rounded bg-slate-800 px-1 font-mono text-slate-300">Tab</kbd>,{" "}
          <kbd className="rounded bg-slate-800 px-1 font-mono text-slate-300">,</kbd> or{" "}
          <kbd className="rounded bg-slate-800 px-1 font-mono text-slate-300">Enter</kbd>{" "}
          to lock it in and jump to the next pick.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <section className="rounded-lg border border-sky-900/60 bg-sky-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
            Blue Side
          </h2>
          <div className="flex flex-col gap-3">
            {teams && Object.keys(teams.teams).length > 0 && (
              <TeamPicker side="blue" value={blueOrg} teams={teams} onChange={pickBlueOrg} />
            )}
            {showPlayers && blueOrg && (
              <RosterEditor
                side="blue"
                seats={bluePlayers}
                players={players!}
                onSeat={setBlueSeat}
              />
            )}
            {ROLES.map((role, i) => (
              <div key={role} className="flex flex-col">
                <ChampionPicker
                  role={role}
                  side="blue"
                  seq={i}
                  value={blue[role]}
                  champions={champions}
                  takenElsewhere={allTaken}
                  onChange={(champion) => setBlue((prev) => ({ ...prev, [role]: champion }))}
                />
                <SuggestionLine
                  suggestion={suggestions.blue[role]}
                  side="blue"
                  onPick={(champion) => setBlue((prev) => ({ ...prev, [role]: champion }))}
                />
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-rose-400">
            Red Side
          </h2>
          <div className="flex flex-col gap-3">
            {teams && Object.keys(teams.teams).length > 0 && (
              <TeamPicker side="red" value={redOrg} teams={teams} onChange={pickRedOrg} />
            )}
            {showPlayers && redOrg && (
              <RosterEditor
                side="red"
                seats={redPlayers}
                players={players!}
                onSeat={setRedSeat}
              />
            )}
            {ROLES.map((role, i) => (
              <div key={role} className="flex flex-col">
                <ChampionPicker
                  role={role}
                  side="red"
                  seq={5 + i}
                  value={red[role]}
                  champions={champions}
                  takenElsewhere={allTaken}
                  onChange={(champion) => setRed((prev) => ({ ...prev, [role]: champion }))}
                />
                <SuggestionLine
                  suggestion={suggestions.red[role]}
                  side="red"
                  onPick={(champion) => setRed((prev) => ({ ...prev, [role]: champion }))}
                />
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          data-draft-predict
          disabled={!canPredict}
          onClick={handlePredict}
          className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          Predict Winner
        </button>
        <button
          type="button"
          onClick={swapSides}
          title="Swap blue and red — drafts, teams, and players"
          className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800"
        >
          ⇄ Swap sides
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
              {result.teamContext.crossRegion && (
                <p className="rounded-md border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-300/90">
                  Cross-region matchup ({result.teamContext.blueLeague} vs{" "}
                  {result.teamContext.redLeague}) — like an international event
                  (MSI / Worlds / EWC). Team Elo is calibrated mostly WITHIN
                  each league and only weakly across regions, so this
                  prediction is less reliable: held-out accuracy on
                  international events is ~57% vs ~65% for same-region games,
                  even after up-weighting inter-region games. Treat the edge
                  with extra caution.
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
