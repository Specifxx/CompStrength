export type Role = "TOP" | "JUNGLE" | "MID" | "BOTTOM" | "SUPPORT";

export const ROLES: Role[] = ["TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT"];

export type SampleConfidence = "low" | "medium" | "high";

export interface ChampionRating {
  primaryRole: Role;
  proGames: number;
  proWinRate: number;
  soloGames: number;
  soloWinRate: number;
  blendedWinRate: number;
  strengthScore: number;
  pickRate: number;
  banRate: number;
  sampleConfidence: SampleConfidence;
}

export interface ChampionRatingsFile {
  generatedAt: string;
  patch: string;
  params: {
    patchHalfLifeDays: number;
    soloQueueWeight: number;
    priorGames: number;
    proWindowDays: number;
  };
  globalMean: number;
  champions: Record<string, ChampionRating>;
  // Added alongside synergy.json; absent on older snapshots.
  patchesUsed?: string[];
}

export interface ModelFile {
  version: number;
  trainedAt: string;
  trainingGames: number;
  coefficients: {
    scoreDiffWeight: number;
    blueSideBias: number;
    intercept: number;
    // Added alongside synergy.json. May be absent on older model.json
    // snapshots — treat as `?? 0` wherever this is consumed.
    synergyWeight?: number;
    matchupWeight?: number;
    // Meta-presence (pickRate + banRate) feature weight. Absent on older
    // snapshots — treat as `?? 0`.
    presenceWeight?: number;
  };
  metrics: {
    logLoss: number;
    accuracy: number;
    baselineAccuracy: number;
    note?: string;
  };
}

export interface SynergyPairStat {
  gamesDecayed: number;
  residual: number;
}

export interface SynergyFile {
  generatedAt: string;
  patch: string;
  patchesUsed: string[];
  params: {
    synergyPriorGames: number;
    matchupPriorGames: number;
  };
  // key: two champion names sorted alphabetically, joined with "|"
  synergy: Record<string, SynergyPairStat>;
  // key: "{attackerChampion}>{opponentChampion}", directional
  matchup: Record<string, SynergyPairStat>;
}

export interface BacktestCalibrationBucket {
  bucket: string;
  predictedMean: number;
  actualWinRate: number;
  count: number;
}

/** Held-out accuracy for one segment (a league or a patch). */
export interface BacktestSegment {
  name: string;
  testGames: number;
  accuracy: number;
  baselineAccuracy: number;
  logLoss: number;
  blueWinRate: number;
}

/** One patch's share of the training data + its patch-recency weight. */
export interface CompositionPatch {
  name: string;
  games: number;
  recencyWeight: number;
}

/** One league's share of the training data. */
export interface CompositionLeague {
  name: string;
  games: number;
}

export interface BacktestReportFile {
  generatedAt: string;
  folds: number;
  testGames: number;
  metrics: {
    accuracy: number;
    logLoss: number;
    brierScore: number;
    baselineAccuracy: number;
    coinFlipLogLoss: number;
  };
  calibration: BacktestCalibrationBucket[];
  // Composition of the data the model is built on, and held-out accuracy
  // broken down by patch/league. Optional so older reports still type-check.
  dataComposition?: {
    totalGames: number;
    byPatch: CompositionPatch[];
    byLeague: CompositionLeague[];
  };
  breakdowns?: {
    byPatch: BacktestSegment[];
    byLeague: BacktestSegment[];
  };
  note?: string;
}

export interface ChampionListItem extends ChampionRating {
  name: string;
}

export type DraftSide = "blue" | "red";

export interface DraftTeam {
  TOP: string | null;
  JUNGLE: string | null;
  MID: string | null;
  BOTTOM: string | null;
  SUPPORT: string | null;
}

export interface PredictRequest {
  blue: DraftTeam;
  red: DraftTeam;
}

export interface ChampionContribution {
  champion: string;
  role: Role;
  strengthScore: number;
  blendedWinRate: number;
  proWinRate: number;
  proGames: number;
  soloWinRate: number;
  sampleConfidence: SampleConfidence;
}

export interface NotablePair {
  pair: [string, string];
  residual: number;
}

export interface NotablePairs {
  synergy: NotablePair[];
  matchup: NotablePair[];
}

export interface PredictResponse {
  patch: string;
  blueWinProbability: number;
  redWinProbability: number;
  blueTeamScore: number;
  redTeamScore: number;
  synergyDiff: number;
  matchupDiff: number;
  breakdown: {
    blue: ChampionContribution[];
    red: ChampionContribution[];
  };
  modelMetrics: ModelFile["metrics"];
  notablePairs: NotablePairs;
}
