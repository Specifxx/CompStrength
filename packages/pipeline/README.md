# compstrength-pipeline

Python ETL + model-fitting pipeline for [CompStrength](https://github.com/), a
hobby League of Legends esports analytics site. It turns raw pro-match data
(Oracle's Elixir, cross-checked against Leaguepedia) plus solo-queue champion
win rates into two static JSON files that the Next.js frontend serves
directly: `data/champion_ratings.json` and `data/model.json`.

## Quick start

```bash
cd packages/pipeline
pip install -e .
python -m pytest tests -q
python -m compstrength_pipeline.build
```

By default `build` reads the bundled synthetic fixtures under
`tests/fixtures/` and writes to `../../data/` (i.e. `/data` at the repo
root). This makes the whole pipeline runnable fully offline, which is also
how it's validated in CI/sandboxed environments where oracleselixir.com,
lol.fandom.com, and lolalytics.com are not reachable.

To point it at live data instead (e.g. from a GitHub Actions workflow with
open network egress):

```bash
python -m compstrength_pipeline.build \
  --oracles-elixir-url https://oracleselixir-downloadable-match-data.s3.us-east-2.amazonaws.com/2026_LoL_esports_match_data_from_OraclesElixir.csv \
  --soloqueue-fixture path/to/live_or_cached_soloqueue.json
```

All of the above are also configurable via environment variables
(`ORACLES_ELIXIR_URL`, `ORACLES_ELIXIR_PATH`, `SOLOQUEUE_FIXTURE_PATH`,
`COMPSTRENGTH_PATCH`, `COMPSTRENGTH_OUTPUT_DIR`) — see `build.py --help`.

## What each source does

- **`sources/oracles_elixir.py`** — the primary data source. Loads Oracle's
  Elixir's wide-format CSV (one row per player per game, plus 2 team-summary
  rows per game carrying bans/picks) and normalizes it into a canonical
  per-player-game table plus a normalized bans table. Works against a local
  CSV path (fixture or downloaded snapshot) or a live URL.
- **`sources/leaguepedia.py`** — a cross-check/supplement, querying
  Leaguepedia's Cargo API (`ScoreboardGames`, `ScoreboardPlayers`,
  `PicksAndBansS7`) for draft/side data. Not required for the pipeline to
  run; useful for spotting data-entry discrepancies against Oracle's Elixir.
- **`sources/soloqueue.py`** — provides champion win rates from ranked solo
  queue, used as an informative Bayesian prior in `features.py` (pro sample
  sizes are tiny; solo queue is not). Two implementations:
  - `LolalyticsSoloQueueSource` — best-effort live adapter for lolalytics.com.
  - `StaticSoloQueueSource` — reads a local JSON fixture; used by default,
    by tests, and for any offline/demo run.

## Live data vs. fixtures

Every "live" fetch function in `sources/` is wrapped so that network
failures raise a clear `DataSourceUnavailableError` explaining that egress
may be blocked, instead of a raw connection traceback. None of the fetcher
*code* is network-sandbox-aware or special-cased — it's written to work
unmodified in an environment with open egress (e.g. GitHub Actions). This
sandbox's network egress to oracleselixir.com, lol.fandom.com, and
lolalytics.com is blocked, so all validation here uses the bundled synthetic
fixtures in `tests/fixtures/` (see `tests/fixtures/README.md`: **fixture data
is synthetic, not real match history**).

## ToS / rate-limit caveats for the solo-queue source

`lolalytics.com` does not publish an official public API — the endpoint used
by `LolalyticsSoloQueueSource` is a reverse-engineered internal JSON endpoint
also used by several hobby projects (e.g. PyPI's `lolalytics-api`,
`khorn89/LolAlytics.py`). Treat it as best-effort:

- No documented ToS for third-party programmatic use; it can change or
  break without notice.
- Make at most one request per `get_champion_winrates(patch)` call — do not
  hammer the endpoint in a loop across many patches/ranks/regions.
- A more defensible (but much heavier) alternative is the official Riot
  Developer API: pull `match-v5` data at scale yourself and aggregate win
  rates. Fully ToS-compliant and stable, but requires your own
  crawling/aggregation infrastructure and is subject to Riot's rate limits.
- OP.GG's official MCP server (`opgginc/opgg-mcp`) is another
  officially-sanctioned option worth evaluating as a secondary/fallback
  adapter, though it's an MCP interface rather than a plain REST/JSON
  endpoint, so it isn't implemented here.

## Package layout

```
compstrength_pipeline/
  config.py         hyperparameters (PATCH_HALF_LIFE_DAYS, SOLO_QUEUE_WEIGHT, ...) + source URLs
  sources/
    oracles_elixir.py   primary data source (fetch + normalize + extract bans)
    leaguepedia.py       cross-check/supplement (Cargo API)
    soloqueue.py         solo-queue win rate prior (lolalytics + static fixture)
  etl.py            cleaning/validation (drop incomplete games, parse dates, standardize casing)
  features.py       empirical-Bayes shrinkage math (the core stats logic)
  train_model.py    logistic regression: score-diff -> P(blue wins)
  build.py          CLI entrypoint: ETL -> features -> train_model -> write JSON
```

## Output files

`build.py` writes exactly two files to `/data` at the repo root:
`champion_ratings.json` and `model.json`. See `compstrength_pipeline/build.py`
docstrings and `apps/web/lib/types.ts` for the exact schema (the two are kept
in sync — the frontend types were authored against this same schema).
