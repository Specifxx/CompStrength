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
          into a logistic model of blue-side win probability &mdash; and,
          when you optionally select the two teams, adds team Elo, per-player
          Elo, and player-champion comfort from the inferred starting fives.
          The numbers below come from holding out games the model never
          trained on.
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
                  Team strength carries most of the predictable signal in pro
                  play; the draft refines it. &ldquo;Teams + draft&rdquo; uses
                  three history signals that all ride on the team selection:
                  team Elo over the game history, per-player Elo of the five
                  starters (tracks roster moves the team rating smooths over),
                  and each player&apos;s record on the champion they&apos;re
                  drafting (comfort picks). Select both teams on the draft
                  builder to get the top row&apos;s model &mdash; the starting
                  five is pre-filled from each team&apos;s most recent game and
                  you can edit any seat to swap in a substitute.
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  <span className="text-slate-300">
                    Team Elo scores how decisively a game was won, not just
                    who won it.
                  </span>{" "}
                  Classic Elo moves ratings by{" "}
                  <span className="font-mono">K &times; (result &minus; expected)</span>,
                  which treats a 12-minute demolition and a 45-minute base race
                  as the same event &mdash; throwing away the most informative
                  thing about a pro game. Each win now moves ratings in
                  proportion to how decisive it was, measured by the winner&apos;s
                  gold lead as a fraction of the loser&apos;s gold (relative, so
                  it compares fairly across game lengths), with diminishing
                  returns on huge margins and a damping term so favourites who
                  were *expected* to stomp don&apos;t run away. Held out on the
                  walk-forward backtest this lifted current-season accuracy{" "}
                  <span className="text-emerald-400">65.7% &rarr; 66.2%</span>{" "}
                  and log-loss{" "}
                  <span className="text-emerald-400">0.622 &rarr; 0.618</span>.
                  The margin only ever affects the post-game update &mdash;
                  every prediction still uses strictly pre-game ratings.
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  <span className="text-slate-300">
                    A stronger-looking version of this was rejected.
                  </span>{" "}
                  Replacing the win/loss outcome entirely with a continuous
                  dominance score scored better on the pooled metric
                  (66.4%/0.613) &mdash; but it compressed the rating spread by
                  two-thirds, which flattens Elo&apos;s expected-score term and
                  quietly removes the opponent-strength correction. The ratings
                  degenerated into &ldquo;average gold margin against whoever
                  you happened to play&rdquo;: minor-league and academy sides
                  floated to the top of the table (majors held 1 of the top 10
                  slots, down from 6), and accuracy on cross-region games
                  actually <span className="text-rose-400">dropped</span> 61.9%
                  &rarr; 60.7%. The pooled number hid it because international
                  games are a small slice of the total. The shipped version
                  moves the ratings the other way, making them{" "}
                  <span className="text-slate-300">more</span> separated than
                  before (spread +10%).
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  <span className="text-slate-300">
                    Aren&apos;t multi-season team ratings stale, since rosters
                    change?
                  </span>{" "}
                  We tested exactly that. Rebuilding team strength from the
                  current season ONLY (resetting every rating at the year
                  boundary) scores <span className="text-rose-400">62.9%</span>{" "}
                  &mdash; 2.3 points WORSE than keeping prior-season history.
                  The reason is subtle: teams churn, but{" "}
                  <span className="text-slate-300">players don&apos;t</span>.
                  When both team and player ratings are reset, accuracy craters;
                  keep the players&apos; individual histories and it recovers
                  almost entirely. Last season stays informative because the
                  same people are still playing &mdash; which is the whole point
                  of the per-player Elo feature. A mild discount on old team
                  ratings is optimal (the shipped carryover keeps 70% of a
                  team&apos;s prior-season deviation), but throwing last season
                  away is strictly worse.
                </p>
              </div>
            )}
            {report.note && (
              <p className="mt-4 text-xs text-slate-500">{report.note}</p>
            )}
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
              Why draft-only sits near 55% (what we tested)
            </h2>
            <div className="flex flex-col gap-2 text-sm text-slate-400">
              <p>
                It&apos;s tempting to think a hard counter-pick (Renekton into
                Shen, Ashe into Teemo) should let a draft-only model predict
                much better than ~55%. We tested that directly, with a
                leak-free walk-forward harness (each model trained only on
                games strictly before the games it&apos;s scored on), and none
                of these beat the shipped model on both accuracy and
                calibration:
              </p>
              <ul className="ml-4 list-disc space-y-1 text-xs text-slate-500">
                <li>
                  <span className="text-slate-300">Solo-queue counter-picks.</span>{" "}
                  A dense champion-vs-champion lane-matchup prior from ~9,900
                  solo-queue pairs (emerald+, far more than pro data ever
                  sees). Result: no change (54.7% &rarr; 54.7%). Pro lane
                  outcomes are dominated by team coordination, not the solo
                  matchup.
                </li>
                <li>
                  <span className="text-slate-300">Off-meta / off-role picks.</span>{" "}
                  Flagging picks played out of their usual role (the
                  &ldquo;Ashe top&rdquo; signal). Nudged accuracy up ~0.3pp but
                  hurt calibration &mdash; directionally real, too noisy to
                  trust as a probability.
                </li>
                <li>
                  <span className="text-slate-300">More history.</span> Training
                  on 3&ndash;4 seasons instead of 2. No gain: recency decay
                  already fades old seasons to near-zero weight.
                </li>
                <li>
                  <span className="text-slate-300">League effects &amp; pick
                  order.</span> Per-league blue-side bias, region-local metas,
                  and pro pick-order (who counter-picked whom). All within
                  noise or accuracy-up/calibration-down.
                </li>
              </ul>
              <p>
                <span className="text-slate-300">
                  The draft-only mode now has its own dedicated fit.
                </span>{" "}
                Previously the no-teams prediction reused the full
                model&apos;s coefficients with the team terms zeroed &mdash;
                but those weights were fit <em>with</em> team features
                present, and without them the pairwise-synergy term (which is
                estimated in-sample) gets over-trusted. Measured on the
                walk-forward harness, adding synergy to a draft-only fit makes
                it <span className="text-rose-400">worse</span> than champion
                strength alone (55.7% vs 56.0% current-season), while the
                lane-matchup term helps. The shipped draft-only fit therefore
                uses champion strength + lane matchups with synergy excluded:{" "}
                <span className="text-emerald-400">56.2% vs 56.0%</span>{" "}
                current-season and{" "}
                <span className="text-emerald-400">55.1% vs 54.6%</span>{" "}
                overall against the old zeroed-coefficients behaviour, with
                better calibration on both slices. (The draft builder&apos;s
                synergy-pick suggestion still ranks through the full
                model&apos;s synergy weight &mdash; it&apos;s a relative
                comparison among candidates, not a calibrated probability.)
                We also tested a champion &ldquo;dominance&rdquo; rating
                &mdash; margin-of-victory averaged per champion, the
                champion-level analogue of the team-Elo margin weighting.
                Rejected: log-loss identical to four decimals in every
                configuration; dominance alone predicts 55.2%, nearly as well
                as win rates do, confirming the two measure the same
                underlying champion strength.
              </p>
              <p>
                The honest conclusion: at the pro level, which champions get
                drafted is a genuinely weak predictor once you can&apos;t see
                who&apos;s piloting them. That&apos;s exactly why the
                &ldquo;teams + players&rdquo; model jumps ~10 points to ~66%
                &mdash; the predictable signal lives in team and player
                identity, not the champion select screen. Disaster drafts are
                real but rare (a handful per season), so even predicting every
                one perfectly moves aggregate accuracy under a point.
              </p>
              <p>
                <span className="text-slate-300">
                  Should it train on the current season only?
                </span>{" "}
                We measured that too. Training on this season alone (walk-forward
                within it) scores <span className="text-rose-400">63.7%</span>{" "}
                with teams, ~2 points BELOW the shipped model that also uses last
                season as history &mdash; the early-season games starve without
                it. More recent-weighted data doesn&apos;t beat the current
                recency decay; less data just loses signal.
              </p>
              <p>
                <span className="text-slate-300">
                  International events (MSI / Worlds / EWC) are the hardest.
                </span>{" "}
                Held-out accuracy on cross-region events runs well below
                regional play &mdash; ~57% aggregate (EWC 66%, Worlds 58%, MSI
                ~52% on a tiny 80-game sample) vs 65% inside a single league.
                The cause is structural: team Elo only bridges regions through
                the handful of inter-region games, so when the best of two
                regions meet the rating gap is genuinely uncertain. We help it
                where we can &mdash; inter-region games (an international
                league, or any game where the two teams&apos; home leagues
                differ) move Elo 3&times; as much, since they&apos;re the only
                games that calibrate strength across regions. That lifted
                held-out international accuracy 55.7% &rarr; 56.5% and improved
                its calibration (log-loss 0.715 &rarr; 0.714) with no cost to
                regional games &mdash; a real but modest gain; cross-region
                prediction stays fundamentally data-starved. The draft builder
                flags these matchups so you can weight them accordingly.
              </p>
              <p>
                <span className="text-slate-300">
                  What about wombo combos, all-AD/all-AP teams, or a
                  low-damage/all-squishy comp?
                </span>{" "}
                This is a real gap: the model&apos;s only inter-champion signal
                is pairwise synergy (exactly 2 champions at a time, shrunk
                toward 0) &mdash; there is no feature that looks at the
                5-champion whole. We built the best measurable proxies
                available and tested each: team damage output and tankiness
                (from real per-game damage/tanking stats), damage-share
                concentration (one carry vs a balanced comp), and AD/AP
                balance and engage-champion count (from champion attributes,
                since Oracle&apos;s Elixir has no crowd-control-time column to
                measure a true &ldquo;wombo&rdquo; signal from). Draft-only,
                one candidate (AD/AP lean) showed a small, consistent gain
                across every test fold. But once combined with team and
                player Elo &mdash; the model actually used once you pick
                teams &mdash; that gain vanished into noise (currentSeason
                65.71% &rarr; 65.76%, well within run-to-run variance) while
                overall accuracy and log-loss both got marginally worse.
                Team/player identity is simply too dominant a signal for
                these subtler structural effects to move the needle on real
                pro outcomes. None of these features ship.
              </p>
              <p>
                <span className="text-slate-300">
                  Why not just use Riot&apos;s official Global Power Rankings?
                </span>{" "}
                We tested it properly rather than assuming. The GPR history is
                real and, importantly, not retroactively rewritten &mdash; we
                verified 7,032 overlapping historical observations fetched
                through different query windows and found zero conflicts, so
                using it wouldn&apos;t leak future results into past
                predictions. Added as a feature it changed held-out log-loss
                from 0.6125 to 0.6125: <span className="text-slate-300">nothing</span>.
                The reason is that GPR is itself an Elo-style rating computed
                from the same match results our own Elo already consumes
                &mdash; and ours covers all 612 teams in the data, where GPR
                ranks only ~58 tier-1 teams (about a quarter of games, with no
                academy or regional sides at all). It is a less complete
                version of a signal the model already has.
              </p>
              <p>
                <span className="text-slate-300">
                  What about form, streaks, rest days, or head-to-head records?
                </span>{" "}
                Also tested, also negative. Recent form correlates{" "}
                <span className="text-slate-300">0.75</span> with team Elo
                &mdash; it is a noisier restatement of what the rating already
                encodes. Win/loss streaks actively hurt (&minus;0.5 points of
                current-season accuracy). Days of rest and games-in-the-last-week
                have essentially no relationship with the outcome at all
                (correlation ~0.015). Head-to-head history, series game number,
                playoffs-vs-regular-season, and roster changes all landed inside
                the measurement noise. The honest summary: once you know who is
                playing and how dominant they have been, schedule and momentum
                narratives add nothing measurable.
              </p>
            </div>
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
