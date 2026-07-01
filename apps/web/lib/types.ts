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
    // Team-strength (Elo gap / eloScale) feature weight. Absent on older
    // snapshots — treat as `?? 0`.
    teamEloWeight?: number;
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
  // Companion metrics from the draft-only refit (team feature zeroed) on the
  // SAME held-out games. Present alongside teamFeatureUsed on new reports.
  draftOnlyMetrics?: {
    accuracy: number;
    logLoss: number;
    brierScore: number;
  };
  teamFeatureUsed?: boolean;
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
  // Optional org names (keys into teams.json). When BOTH are provided, the
  // model adds the team-strength (Elo gap) term; when either is missing the
  // gap is 0 ("assume equal teams") and the prediction is draft-only.
  blueTeam?: string | null;
  redTeam?: string | null;
}

/** One professional team's Elo rating snapshot (data/teams.json). */
export interface TeamRating {
  elo: number;
  games: number;
  league: string;
  lastPlayed: string;
}

export interface TeamsFile {
  generatedAt: string;
  patch: string;
  params: {
    eloK: number;
    eloScale: number;
    initialElo: number;
  };
  teams: Record<string, TeamRating>;
}

/** Team-strength inputs actually applied to a prediction. */
export interface TeamContext {
  blueTeam: string;
  redTeam: string;
  blueElo: number;
  redElo: number;
  /** (blueElo - redElo) / eloScale — the model feature value. */
  eloDiff: number;
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
  // Present only when both teams were provided AND found in teams.json —
  // i.e. when the team-strength term actually contributed to the logit.
  teamContext?: TeamContext | null;
}
