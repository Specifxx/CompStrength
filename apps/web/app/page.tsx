import { DraftBuilder } from "@/components/DraftBuilder";
import {
  DataNotReadyError,
  loadBacktestReport,
  loadChampionRatings,
  loadModel,
  loadSynergy,
} from "@/lib/data";
import type {
  BacktestReportFile,
  ChampionListItem,
  ChampionRatingsFile,
  ModelFile,
  SynergyFile,
} from "@/lib/types";

function loadHomeData(): {
  champions: ChampionListItem[];
  ratings: ChampionRatingsFile;
  patch: string;
  model: ModelFile;
  synergy: SynergyFile;
  backtest: BacktestReportFile | null;
} | null {
  try {
    const ratings = loadChampionRatings();
    const champions: ChampionListItem[] = Object.entries(ratings.champions).map(
      ([name, rating]) => ({ name, ...rating }),
    );
    const model = loadModel();
    // synergy.json may not exist yet (pipeline still being extended) —
    // fall back to an empty synergy file so the app still renders and
    // predictMatchup treats every pair as a neutral (0) residual.
    let synergy: SynergyFile;
    try {
      synergy = loadSynergy();
    } catch (err) {
      if (!(err instanceof DataNotReadyError)) throw err;
      synergy = {
        generatedAt: "",
        patch: ratings.patch,
        patchesUsed: [],
        params: { synergyPriorGames: 0, matchupPriorGames: 0 },
        synergy: {},
        matchup: {},
      };
    }
    // The walk-forward backtest report drives the honest, held-out accuracy
    // shown under a prediction (see DraftBuilder). Optional: if it hasn't been
    // generated yet, fall back to null and the footer degrades gracefully.
    let backtest: BacktestReportFile | null;
    try {
      backtest = loadBacktestReport();
    } catch (err) {
      if (!(err instanceof DataNotReadyError)) throw err;
      backtest = null;
    }
    return { champions, ratings, patch: ratings.patch, model, synergy, backtest };
  } catch (err) {
    if (err instanceof DataNotReadyError) return null;
    throw err;
  }
}

export default function HomePage() {
  const data = loadHomeData();

  if (!data) {
    return (
      <div className="mx-auto flex max-w-2xl flex-1 flex-col justify-center gap-4 px-4 text-center">
        <h1 className="text-2xl font-bold">
          Comp<span className="text-sky-400">Strength</span>
        </h1>
        <p className="text-slate-400">
          Champion ratings haven&apos;t been generated yet. Run the data
          pipeline from the repo root:
        </p>
        <pre className="rounded-md bg-slate-900 p-4 text-left text-sm text-slate-300">
          {"pip install -e packages/pipeline\npython -m compstrength_pipeline.build"}
        </pre>
      </div>
    );
  }

  return (
    <DraftBuilder
      champions={data.champions}
      ratings={data.ratings}
      patch={data.patch}
      model={data.model}
      synergy={data.synergy}
      backtest={data.backtest}
    />
  );
}
