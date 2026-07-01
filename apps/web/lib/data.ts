import fs from "node:fs";
import path from "node:path";
import type { ChampionRatingsFile, ModelFile } from "./types";

// The pipeline (packages/pipeline) writes its generated snapshot to
// <repo root>/data/*.json. This app lives at <repo root>/apps/web, so we
// read two directories up. next.config.ts declares these paths under
// outputFileTracingIncludes so Vercel bundles them into the serverless
// function output.
const DATA_DIR = path.join(process.cwd(), "..", "..", "data");

export class DataNotReadyError extends Error {}

function readJson<T>(filename: string): T {
  const filePath = path.join(DATA_DIR, filename);
  if (!fs.existsSync(filePath)) {
    throw new DataNotReadyError(
      `${filename} not found in ${DATA_DIR}. Run the pipeline first: ` +
        `pip install -e packages/pipeline && python -m compstrength_pipeline.build`,
    );
  }
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

let ratingsCache: ChampionRatingsFile | null = null;
let modelCache: ModelFile | null = null;

export function loadChampionRatings(): ChampionRatingsFile {
  if (process.env.NODE_ENV === "production" && ratingsCache) return ratingsCache;
  const data = readJson<ChampionRatingsFile>("champion_ratings.json");
  ratingsCache = data;
  return data;
}

export function loadModel(): ModelFile {
  if (process.env.NODE_ENV === "production" && modelCache) return modelCache;
  const data = readJson<ModelFile>("model.json");
  modelCache = data;
  return data;
}
