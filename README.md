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
patch-recency decay. **A note on MSI 2026 specifically**: as of this
writing MSI 2026 is still in progress (Play-In just concluded, Bracket Stage
runs into mid-July), and this sandbox's network policy blocks the
structured data sources (Oracle's Elixir, Leaguepedia) that would let the
pipeline actually ingest it — search-engine snippets are not a reliable
substitute (results for an in-progress tournament are prone to
speculative/fabricated content, and no full per-game draft data was
recoverable that way). Once deployed with real network access, the daily
refresh workflow will pick up MSI 2026 games automatically the moment
Oracle's Elixir publishes them — nothing else needs to change.

## Data sources

- **[Oracle's Elixir](https://oracleselixir.com/)** — the primary source of
  professional match data (games, results, picks, patches, dates). This is
  the backbone dataset used to compute recent pro-play performance per
  champion and comp.
- **[Leaguepedia Cargo API](https://lol.fandom.com/wiki/Special:CargoTables)**
  — used to cross-check picks/bans and match metadata against Oracle's
  Elixir, to catch gaps or discrepancies in either source.
- **Solo-queue win-rate source** (e.g. a public champion stats aggregator) —
  used as a secondary prior for champions with limited professional sample
  size. The specific provider is an implementation detail of the pipeline and
  may change.

Pipeline fetchers are written to be **pluggable/swappable** — each data
source lives behind its own fetcher module in `packages/pipeline`, so a
source can be replaced (e.g. if a site changes its API or shuts down) without
touching the modeling code downstream.

**A caveat on ToS and rate limits:** community stats sites (Oracle's Elixir,
Leaguepedia, solo-queue aggregators) are typically maintained by volunteers
or small teams, not designed for high-volume automated access. This project
fetches data **at most once a day**, caches what it can, and is intended to
stay a good citizen of those sites' terms of service and rate limits. If
you fork this project, please review the current ToS of whichever sources you
point the pipeline at, and keep request volume low.

## How the model works

At a high level, the pipeline turns raw match history into a single blue-side
win probability estimate via five steps:

1. **Recent-patches-only cutoff.** Only the `NUM_RECENT_PATCHES` most recent
   patches (default 3) that actually appear in the match data are considered
   at all — anything older is excluded outright, not just down-weighted, since
   older patches can reflect a meaningfully different game. Patches are
   ordered by the actual date of play, not by sorting the patch string.
2. **Patch-recency weighting *within* that window.** Older games still matter
   less than newer ones inside the recent-patch window. Each historical game
   is weighted by an exponential decay based on how long ago it was played,
   controlled by a `PATCH_HALF_LIFE_DAYS` hyperparameter — a game from one
   half-life ago counts for half as much as a game played today, so the most
   recent patch dominates the rating.
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
surfaces these numbers. **Today, this runs against a small synthetic fixture
dataset** (see `packages/pipeline/tests/fixtures/README.md`) — the reported
numbers are illustrative of the *methodology*, not a real accuracy claim,
until the pipeline is pointed at real historical Oracle's Elixir data (see
"Local development" below for how to do that).

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
python -m compstrength_pipeline.build
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

- **Side-specific (blue/red) champion win-rate deltas** — some champions may
  perform meaningfully differently on blue vs. red side beyond the generic
  side-advantage term.
- **Backfilling the full historical Oracle's Elixir archive**, rather than
  only a recent rolling window, to better support long-term trend analysis.
- **Per-league model variants** (LCK vs. LPL vs. LEC vs. others), since
  regional metas and playstyles can diverge from the global average.
