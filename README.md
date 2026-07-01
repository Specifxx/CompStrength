# CompStrength

CompStrength is a hobby analytics site that predicts **win probability for a
professional League of Legends draft** — blue side vs. red side, 5 champions
each — based on how those champions and comps have actually performed in
recent professional play, weighted by patch recency, and blended with
solo-queue win rates as a statistical prior. Punch in (or explore) a draft and
get a calibrated blue-side win probability, grounded in real match data rather
than vibes.

## Architecture

CompStrength is intentionally three simple pieces, with no server or database
in between:

```
packages/pipeline (Python)          data/*.json (committed)         apps/web (Next.js on Vercel)
  fetch pro match data      --->      champion_ratings.json  --->    prediction computed
  fetch solo-queue stats               model.json                    entirely in the browser,
  fit model, backtest it               synergy.json                  zero network round-trip
  write JSON + report                  backtest_report.json          (API route also available)
```

1. **`packages/pipeline`** — a Python job that pulls recent professional match
   data (restricted to the last few patches) and solo-queue win rates, fits a
   small statistical model — including champion-pair synergy and lane-matchup
   effects — backtests it, and writes the results as plain JSON files.
2. **`data/champion_ratings.json`**, **`data/model.json`**,
   **`data/synergy.json`**, and **`data/backtest_report.json`** — the *only*
   data store. These files are committed to the repo (not a database), so the
   entire "backend" is version-controlled, diffable, and free to host.
3. **`apps/web`** — a Next.js 16 App Router site deployed on Vercel. The win
   probability is computed **directly in the browser** the instant a draft is
   filled in — the champion ratings, model coefficients, and synergy/matchup
   tables are loaded once and the (pure, dependency-free) scoring function
   runs client-side, so there is no network latency on the "Predict Winner"
   click at all. The same scoring function backs a `/api/predict` route for
   programmatic use, so both paths always agree.

**Why this design:** win-probability math over a few small JSON files is
cheap enough to run instantly in a browser tab, so there's no need to stand
up and pay for a database, a long-running backend, or even a network round
trip per prediction — which matters if you're using this live while watching
a draft. Data freshness is handled by re-running the pipeline on a schedule
and committing the output — Git itself becomes the data store and its
history. This keeps the whole project inside free tiers (Vercel + GitHub
Actions) and easy for a single hobbyist to reason about: if something looks
wrong, the entire "state of the world" is a handful of JSON files you can
open and read.

## How accurate is this, really?

Be honest with yourself about what this tool is and isn't.

Team composition is only **one** input into who wins a professional match.
Player mechanical skill, individual champion mastery, patch-to-patch meta
shifts, scouting and coaching preparation, in-game decision-making, and live
execution (fights, objectives, macro calls) all matter enormously — arguably
far more than draft alone. Two identical drafts piloted by different rosters,
or played a week apart on a slightly different patch, can produce very
different results.

What CompStrength actually gives you is a **calibrated "draft strength" win
probability**: given historical outcomes for these champions/comps in recent
pro play (and, as a fallback signal, in solo queue), how often has a draft
like this side tended to win, all else being equal? Treat it as:

- A **directional signal**, not a guaranteed outcome predictor.
- A tool for exploring **"what-if" drafts** — e.g., "how does swapping this
  champion change blue side's modeled win rate?"
- Not a substitute for actually watching scouting reports, player form, or
  patch notes.

Statistically, the model is a **Bradley-Terry-style additive log-odds
model**: each champion (and side) contributes an additive term to a log-odds
score, the two teams' scores are compared, and the result is calibrated via
**logistic regression** against historical pro game outcomes. This is a
standard, well-understood approach for "strength rating -> win probability"
problems (it's the same family of model behind chess Elo and many sports
analytics tools), but it is still a simplification of a much messier reality.

**Every champion is selectable, in every role.** Champion ratings aren't
limited to whichever champions happened to get pro games in the current
window — the full known champion roster (`compstrength_pipeline/champions.py`,
sourced live from Data Dragon with a bundled static fallback) is always
included. An off-meta pick with zero recent pro games just falls back to its
solo-queue-informed prior instead of being unselectable, so unconventional
picks (a support-role Teemo, whatever) are always modelable, not just the
current meta's staples.

**International events (MSI, Worlds, EWC) count for more.** These pit the
best team from every region against each other on one current patch, which
makes them an unusually concentrated, high-signal sample of how the current
meta actually resolves at the top level — worth more than an equivalent
regional-split game. `PipelineConfig.international_weight_multiplier`
(default `1.5x`) boosts these games' weight in every stat this pipeline
computes (champion ratings, synergy, matchup), on top of the normal
patch-recency decay. MSI/Worlds games that have actually been played (e.g.
the 2026 MSI Play-In stage) flow in automatically through the live data
source below like any other game — no special-casing needed.

## Data sources

- **[Oracle's Elixir](https://oracleselixir.com/)** (`sources/oracles_elixir.py`)
  — the **primary, live** source of professional match data. It publishes one
  bulk CSV per season (thousands of real pro games; ~150 columns incl.
  `gameid`, `date`, `patch`, `league`, `side`, `position`, `champion`,
  `result`, `ban1..5`), refreshed daily, served from a public Google Drive
  folder. The pipeline downloads the current-season file with `gdown` (one
  request, handling Drive's >25MB confirm-token automatically) and validates
  it's the real CSV (size + `gameid` header) rather than an interstitial page.
  This is the robust choice for a daily cron: **a single bulk download has no
  per-request rate limiting** — the reason it's preferred over the Cargo API
  below.
- **[Leaguepedia Cargo API](https://lol.fandom.com/wiki/Special:CargoTables)**
  (`sources/leaguepedia.py`) — an alternative live source (`ScoreboardGames` +
  `ScoreboardPlayers` + `PicksAndBansS7`), kept as a swappable backup. It
  rate-limits aggressively on shared CI IPs (~40 requests per refresh gets
  throttled), which is exactly why Oracle's Elixir's single bulk download is
  the default.
- **Solo-queue win-rate source** — used as a prior for champions with a
  limited professional sample. Not currently wired to a live source (no
  reliable live endpoint confirmed yet — see Roadmap); when running against
  real match data the pipeline uses a neutral 50% prior instead of guessing,
  rather than blending in made-up numbers.

Pipeline fetchers are **pluggable/swappable** — each source lives behind its
own fetcher module in `packages/pipeline`, so a source can be replaced without
touching the modeling code. Select the source with
`python -m compstrength_pipeline.build --source oracles-elixir` (real data;
what the scheduled GitHub Actions refresh uses), `--source leaguepedia` (the
backup live source), or the default `--source fixture` which reads a small,
clearly-labeled **synthetic** dataset (see
`packages/pipeline/tests/fixtures/README.md`) so the pipeline can run fully
offline for local dev/tests.

**A caveat on ToS and rate limits:** Oracle's Elixir and Leaguepedia are
maintained by volunteers, not designed for high-volume automated access. This
project fetches data **at most once a day** (a single bulk download), and is
intended to stay a good citizen of these sites' terms of service and rate
limits. If you fork this project, please review the current ToS of
whichever sources you point the pipeline at, and keep request volume low.

## How the model works

At a high level, the pipeline turns raw match history into a single blue-side
win probability estimate via five steps:

1. **A target sample size, not a patch cutoff.** The pipeline trains on the
   most recent `TARGET_TRAINING_GAMES` games overall (default 1000),
   regardless of which patch they're on — older games aren't excluded just
   for being on an older patch, they're simply weighted down by step 2
   below. This guarantees a statistically meaningful sample size even when
   the last patch or two alone wouldn't have nearly enough games, at the
   cost of a small amount of stale-meta signal from older patches (which the
   decay below keeps small).
2. **Recency weighting — by calendar date *and* by patch.** Each historical
   game's weight is the product of two decays:
   - *Calendar-day decay*: `0.5 ** (days_ago / PATCH_HALF_LIFE_DAYS)` — a game
     one half-life old counts half as much as one played today.
   - *Patch-ordinal decay*: `PATCH_DECAY_BASE ** patch_distance`, where the
     newest patch (by date, never by lexically sorting the patch string —
     "14.10" is newer than "14.2") has distance 0, the previous patch 1, and
     so on. With the default base 0.5 the current patch counts at full weight,
     the previous at half, two-back at a quarter.
   So the **latest patch(es) dominate** the ratings — both because they're the
   most recent *dates* and because they're the closest *patches* — while older
   patches still contribute a shrinking-but-nonzero amount. International
   events (MSI/Worlds) get an additional weight multiplier on top.
3. **Empirical-Bayes shrinkage toward solo queue.** Many champions have a
   small sample of professional games on the current patch. Rather than
   trusting a noisy small-sample pro win rate outright, the pipeline blends
   it with the champion's solo-queue win rate as a prior, shrinking harder
   toward the prior when the pro sample is small and trusting the pro data
   more as the sample grows. This is controlled by two hyperparameters:
   `SOLO_QUEUE_WEIGHT` (how much overall weight the solo-queue prior gets
   relative to observed pro games) and `PRIOR_GAMES` (the effective sample
   size, in pro games, at which the prior and observed pro data carry roughly
   equal weight).
4. **Additive log-odds (Bradley-Terry-style) rating, plus synergy and
   matchup history.** Each champion gets a learned strength rating in
   log-odds space (from step 3). On top of that, the pipeline mines the same
   match history for two more historical signals, each shrunk toward "no
   effect" the same empirical-Bayes way when sample size is small:
   - **Synergy** — how much better or worse a *specific pair* of champions on
     the same team performs together, beyond what their individual ratings
     already predict.
   - **Matchup** — how a champion's team has historically fared specifically
     when that champion faces a given opposing champion in the same role
     (a coarse proxy for lane-matchup history).
   A side's total score is its five champions' ratings, plus the synergy
   terms for all pairs on that side, plus the matchup terms for each lane.
5. **Logistic calibration.** The blue-minus-red score differential (across
   all three signals above) is passed through a logistic regression fit
   against actual historical pro game outcomes, which maps it to a final,
   calibrated blue-side win probability (a number between 0 and 1) — and also
   corrects for blue side's known small structural advantage (first pick/ban,
   dragon side, etc.).

The fitted champion ratings and calibration parameters are exactly what's
written to `data/champion_ratings.json`, `data/synergy.json`, and
`data/model.json`.

## Backtesting

Because "how accurate is this, really?" deserves a real answer, not just a
promise, the pipeline includes a **walk-forward backtest**
(`compstrength_pipeline/backtest.py`, run automatically at the end of every
`python -m compstrength_pipeline.build`, or standalone via
`python -m compstrength_pipeline.backtest`). It's chronological, not random:
for each fold, the *entire* pipeline (ratings, synergy/matchup tables, model
fit) is recomputed using only games strictly before that fold's start date,
then evaluated against the held-out games that actually happened after —
exactly mirroring how the model would have been used in real time, with no
leakage from the future. It reports accuracy, log-loss, Brier score, and a
calibration table (predicted win probability vs. actual win rate, bucketed),
against a coin-flip and majority-class baseline, and writes
`data/backtest_report.json`. The live site's [`/methodology`](#) page
surfaces these numbers. Run it against real data with
`python -m compstrength_pipeline.build --source leaguepedia` (what the
scheduled refresh does); the default local/test run uses the small
synthetic fixture, in which case treat the numbers as illustrative of the
*methodology*, not a real accuracy claim.

## Local development

### Frontend (Next.js site)

```bash
cd apps/web
npm install
npm run dev
```

This starts the site at `http://localhost:3000`, reading whatever is
currently in `data/champion_ratings.json` and `data/model.json` at the repo
root.

### Pipeline (data refresh)

```bash
pip install -e packages/pipeline
python -m compstrength_pipeline.build                    # bundled synthetic fixture (offline)
python -m compstrength_pipeline.build --source leaguepedia  # real, live data
```

This fetches the latest data, refits the model, and overwrites
`data/champion_ratings.json` and `data/model.json`. Run it, then restart (or
just refresh, thanks to Next.js's dev reload) the frontend to see updated
predictions locally.

## Deployment

This is the important part — follow these steps in order.

1. **Push this repo to GitHub.**
   If you're starting from this repo as-is, it's already on GitHub — nothing
   to do. If you're forking or starting fresh, create a new GitHub repo and
   push this project to it:
   ```bash
   git remote add origin https://github.com/<you>/CompStrength.git
   git push -u origin main
   ```

2. **Import the project on Vercel.**
   - Go to [vercel.com](https://vercel.com) and sign in (GitHub login is
     easiest).
   - Click **Add New... → Project**.
   - Select/import this GitHub repository.
   - Because this is a monorepo, under **Configure Project → Root Directory**,
     click **Edit** and set it to `apps/web`.
   - Leave the **Build Command** and **Output Directory** as their Next.js
     defaults (`next build` / `.next`) — Vercel auto-detects Next.js once the
     root directory is set correctly.
   - Click **Deploy**.

3. **No environment variables or database are required.** The MVP reads
   `data/champion_ratings.json` and `data/model.json` directly from the
   repo at build/request time — there's nothing to configure in Vercel's
   Environment Variables tab for basic functionality.

4. **Enable the GitHub Actions data-refresh workflow.**
   There's nothing extra to configure — once
   `.github/workflows/refresh-data.yml` is merged into your repository's
   default branch, GitHub will automatically run it on the schedule defined
   in the workflow (and it's always available to trigger manually from the
   **Actions** tab via "Run workflow"). The workflow uses the repository's
   default `GITHUB_TOKEN`, which — combined with the `permissions:
   contents: write` set in the workflow file — is sufficient to commit and
   push back to the repo. No extra secrets or personal access tokens are
   needed for this same-repo case.

5. **Let it auto-deploy.**
   Every time the workflow refreshes any of the `data/*.json` files and
   pushes that commit, Vercel — which watches your default branch — will
   automatically kick off a new deployment with the fresh data. There's
   nothing to click; this happens on its own.

6. **Custom domain (optional).**
   In your Vercel project, go to **Settings → Domains**, add your domain, and
   follow Vercel's instructions to point its DNS (A/CNAME records) at Vercel.

7. **On cost:** Vercel's free (Hobby) tier comfortably covers a hobby-scale
   site like this one — there's no database to outgrow, the JSON data files
   are small, and traffic for a niche esports analytics tool is unlikely to
   come close to free-tier limits.

## Roadmap / future ideas

- **A working live solo-queue source.** `LolalyticsSoloQueueSource` exists
  but its endpoint/query params aren't confirmed correct yet; until then,
  real-data runs use a neutral 50% prior instead (see Data sources above).
- **Champion-name canonicalization against Data Dragon**, so off-meta picks
  and irregular spellings (LeBlanc, Wukong/MonkeyKing, Nunu & Willump) join
  cleanly across the games, roster, and any future solo-queue data.
- **Side-specific (blue/red) champion win-rate deltas** — some champions may
  perform meaningfully differently on blue vs. red side beyond the generic
  side-advantage term.
- **Backfilling further into historical data**, rather than only the most
  recent `TARGET_TRAINING_GAMES`, to better support long-term trend analysis.
- **Per-league model variants** (LCK vs. LPL vs. LEC vs. others), since
  regional metas and playstyles can diverge from the global average.
