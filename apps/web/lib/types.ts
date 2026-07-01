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
}

export interface ModelFile {
  version: number;
  trainedAt: string;
  trainingGames: number;
  coefficients: {
    scoreDiffWeight: number;
    blueSideBias: number;
    intercept: number;
  };
  metrics: {
    logLoss: number;
    accuracy: number;
    baselineAccuracy: number;
    note?: string;
  };
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

export interface PredictResponse {
  patch: string;
  blueWinProbability: number;
  redWinProbability: number;
  blueTeamScore: number;
  redTeamScore: number;
  breakdown: {
    blue: ChampionContribution[];
    red: ChampionContribution[];
  };
  modelMetrics: ModelFile["metrics"];
}
