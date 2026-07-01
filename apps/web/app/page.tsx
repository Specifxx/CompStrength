import { DraftBuilder } from "@/components/DraftBuilder";
import {
  DataNotReadyError,
  loadChampionRatings,
  loadModel,
  loadSynergy,
} from "@/lib/data";
import type {
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
    return { champions, ratings, patch: ratings.patch, model, synergy };
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
    />
  );
}
