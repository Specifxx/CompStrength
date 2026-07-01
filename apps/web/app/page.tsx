import { DraftBuilder } from "@/components/DraftBuilder";
import { DataNotReadyError, loadChampionRatings } from "@/lib/data";
import type { ChampionListItem } from "@/lib/types";

function loadHomeData(): { champions: ChampionListItem[]; patch: string } | null {
  try {
    const ratings = loadChampionRatings();
    const champions: ChampionListItem[] = Object.entries(ratings.champions).map(
      ([name, rating]) => ({ name, ...rating }),
    );
    return { champions, patch: ratings.patch };
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

  return <DraftBuilder champions={data.champions} patch={data.patch} />;
}
