import {
  ROLES,
  type ChampionContribution,
  type ChampionRatingsFile,
  type DraftTeam,
  type ModelFile,
  type PredictResponse,
  type Role,
} from "./types";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class UnknownChampionError extends Error {
  constructor(public champion: string) {
    super(`Unknown champion: "${champion}"`);
  }
}

export class IncompleteDraftError extends Error {
  constructor(side: string, missingRole: Role) {
    super(`${side} side is missing a champion in the ${missingRole} role`);
  }
}

function buildSide(
  side: "blue" | "red",
  team: DraftTeam,
  ratings: ChampionRatingsFile,
): ChampionContribution[] {
  return ROLES.map((role) => {
    const champion = team[role];
    if (!champion) throw new IncompleteDraftError(side, role);
    const rating = ratings.champions[champion];
    if (!rating) throw new UnknownChampionError(champion);
    return {
      champion,
      role,
      strengthScore: rating.strengthScore,
      blendedWinRate: rating.blendedWinRate,
      proWinRate: rating.proWinRate,
      proGames: rating.proGames,
      soloWinRate: rating.soloWinRate,
      sampleConfidence: rating.sampleConfidence,
    };
  });
}

export function predictMatchup(
  blue: DraftTeam,
  red: DraftTeam,
  ratings: ChampionRatingsFile,
  model: ModelFile,
): PredictResponse {
  const blueBreakdown = buildSide("blue", blue, ratings);
  const redBreakdown = buildSide("red", red, ratings);

  const blueTeamScore = blueBreakdown.reduce((sum, c) => sum + c.strengthScore, 0);
  const redTeamScore = redBreakdown.reduce((sum, c) => sum + c.strengthScore, 0);

  const { scoreDiffWeight, blueSideBias, intercept } = model.coefficients;
  const logit =
    intercept + scoreDiffWeight * (blueTeamScore - redTeamScore) + blueSideBias;
  const blueWinProbability = sigmoid(logit);

  return {
    patch: ratings.patch,
    blueWinProbability,
    redWinProbability: 1 - blueWinProbability,
    blueTeamScore,
    redTeamScore,
    breakdown: { blue: blueBreakdown, red: redBreakdown },
    modelMetrics: model.metrics,
  };
}
