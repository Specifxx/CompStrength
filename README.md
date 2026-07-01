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
  fetch pro match data      --->      champion_ratings.json   --->    API route computes win %
  fetch solo-queue stats               model.json                    at request time from JSON
  fit model, write JSON                                              static-ish pages read JSON
```

1. **`packages/pipeline`** — a Python job that pulls recent professional match
   data and solo-queue win rates, fits a small statistical model, and writes
   the results as two plain JSON files.
2. **`data/champion_ratings.json`** and **`data/model.json`** — the *only*
   data store. These files are committed to the repo (not a database), so the
   entire "backend" is version-controlled, diffable, and free to host.
3. **`apps/web`** — a Next.js 14 App Router site deployed on Vercel. A Next.js
   API route reads the two JSON files at request time and computes the
   win-probability math directly in TypeScript — no external API calls, no
   database queries, no persistent server process.

**Why this design:** win-probability math over a couple of small JSON files
is cheap enough to do per-request in a serverless function, so there's no
need to stand up and pay for a database or a long-running backend. Data
freshness is handled by re-running the pipeline on a schedule and committing
the output — Git itself becomes the data store and its history. This keeps
the whole project inside free tiers (Vercel + GitHub Actions) and easy for a
single hobbyist to reason about: if something looks wrong, the entire "state
of the world" is two JSON files you can open and read.

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
win probability estimate via four steps:

1. **Patch-recency weighting.** Older games matter less, since champion
   balance and the overall meta shift patch to patch. Each historical game is
   weighted by an exponential decay based on how long ago its patch was
   played, controlled by a `PATCH_HALF_LIFE_DAYS` hyperparameter — a game
   from one half-life ago counts for half as much as a game played today.
2. **Empirical-Bayes shrinkage toward solo queue.** Many champions have a
   small sample of professional games on the current patch. Rather than
   trusting a noisy small-sample pro win rate outright, the pipeline blends
   it with the champion's solo-queue win rate as a prior, shrinking harder
   toward the prior when the pro sample is small and trusting the pro data
   more as the sample grows. This is controlled by two hyperparameters:
   `SOLO_QUEUE_WEIGHT` (how much overall weight the solo-queue prior gets
   relative to observed pro games) and `PRIOR_GAMES` (the effective sample
   size, in pro games, at which the prior and observed pro data carry roughly
   equal weight).
3. **Additive log-odds (Bradley-Terry-style) rating.** Each champion gets a
   single learned strength rating in log-odds space. A team's (side's) total
   strength is the sum of its five champions' ratings; the difference between
   blue side's and red side's totals is the raw signal for who's favored.
4. **Logistic calibration.** That raw log-odds difference is passed through a
   logistic regression fit against actual historical pro game outcomes, which
   maps it to a final, calibrated blue-side win probability (a number between
   0 and 1) — and also corrects for blue side's known small structural
   advantage (first pick/ban, dragon side, etc.).

The fitted champion ratings and calibration parameters are exactly what's
written to `data/champion_ratings.json` and `data/model.json`.

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
   Every time the workflow refreshes `data/champion_ratings.json` and/or
   `data/model.json` and pushes that commit, Vercel — which watches your
   default branch — will automatically kick off a new deployment with the
   fresh data. There's nothing to click; this happens on its own.

6. **Custom domain (optional).**
   In your Vercel project, go to **Settings → Domains**, add your domain, and
   follow Vercel's instructions to point its DNS (A/CNAME records) at Vercel.

7. **On cost:** Vercel's free (Hobby) tier comfortably covers a hobby-scale
   site like this one — there's no database to outgrow, the JSON data files
   are small, and traffic for a niche esports analytics tool is unlikely to
   come close to free-tier limits.

## Roadmap / future ideas

- Pairwise champion **synergy** adjustments (some champions perform better or
  worse specifically alongside certain teammates, beyond their individual
  rating).
- **Role-specific matchup effects**, e.g. modeling lane matchups (top vs top,
  mid vs mid) rather than only whole-team aggregate strength.
- **Side-specific (blue/red) champion win-rate deltas** — some champions may
  perform meaningfully differently on blue vs. red side beyond the generic
  side-advantage term.
- **Backfilling the full historical Oracle's Elixir archive**, rather than
  only a recent rolling window, to better support long-term trend analysis.
- **Per-league model variants** (LCK vs. LPL vs. LEC vs. others), since
  regional metas and playstyles can diverge from the global average.
