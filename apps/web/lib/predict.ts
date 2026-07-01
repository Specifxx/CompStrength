import {
  ROLES,
  type ChampionContribution,
  type ChampionRatingsFile,
  type DraftTeam,
  type ModelFile,
  type NotablePair,
  type PredictResponse,
  type Role,
  type SynergyFile,
  type TeamContext,
  type TeamsFile,
} from "./types";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Key into synergy.synergy: two champion names sorted alphabetically, joined with "|". */
export function synergyKey(a: string, b: string): string {
  return [a, b].sort((x, y) => x.localeCompare(y)).join("|");
}

/** Key into synergy.matchup: directional, "{attacker}>{opponent}". */
export function matchupKey(a: string, b: string): string {
  return `${a}>${b}`;
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

/** Sum of synergy.synergy[...].residual over all 10 unordered pairs among 5 champions. */
function teamSynergySum(champions: string[], synergy: SynergyFile): number {
  let sum = 0;
  for (let i = 0; i < champions.length; i++) {
    for (let j = i + 1; j < champions.length; j++) {
      const key = synergyKey(champions[i], champions[j]);
      sum += synergy.synergy[key]?.residual ?? 0;
    }
  }
  return sum;
}

function topNotablePairs(
  pairs: { pair: [string, string]; residual: number }[],
  n: number,
): NotablePair[] {
  return pairs
    .filter((p) => p.residual !== 0)
    .sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual))
    .slice(0, n);
}

/** Optional team-strength inputs for {@link predictMatchup}. */
export interface TeamOptions {
  blueTeam?: string | null;
  redTeam?: string | null;
  teams?: TeamsFile | null;
}

/** Resolve the team-strength (Elo gap) term. Both-or-nothing: the term only
 *  applies when BOTH org names are given and found in teams.json — a single
 *  known team vs an unspecified opponent would silently be treated as
 *  "vs a perfectly average team", which is misleading rather than neutral. */
function resolveTeamContext(options: TeamOptions | undefined): TeamContext | null {
  if (!options?.teams || !options.blueTeam || !options.redTeam) return null;
  const blueRating = options.teams.teams[options.blueTeam];
  const redRating = options.teams.teams[options.redTeam];
  if (!blueRating || !redRating) return null;
  const scale = options.teams.params.eloScale || 400;
  return {
    blueTeam: options.blueTeam,
    redTeam: options.redTeam,
    blueElo: blueRating.elo,
    redElo: redRating.elo,
    eloDiff: (blueRating.elo - redRating.elo) / scale,
  };
}

export function predictMatchup(
  blue: DraftTeam,
  red: DraftTeam,
  ratings: ChampionRatingsFile,
  model: ModelFile,
  synergy: SynergyFile,
  teamOptions?: TeamOptions,
): PredictResponse {
  const blueBreakdown = buildSide("blue", blue, ratings);
  const redBreakdown = buildSide("red", red, ratings);

  const blueTeamScore = blueBreakdown.reduce((sum, c) => sum + c.strengthScore, 0);
  const redTeamScore = redBreakdown.reduce((sum, c) => sum + c.strengthScore, 0);

  const blueChampions = blueBreakdown.map((c) => c.champion);
  const redChampions = redBreakdown.map((c) => c.champion);

  const synergyDiff =
    teamSynergySum(blueChampions, synergy) - teamSynergySum(redChampions, synergy);

  const notableSynergyCandidates: { pair: [string, string]; residual: number }[] = [];
  for (let i = 0; i < blueChampions.length; i++) {
    for (let j = i + 1; j < blueChampions.length; j++) {
      const a = blueChampions[i];
      const b = blueChampions[j];
      const residual = synergy.synergy[synergyKey(a, b)]?.residual ?? 0;
      if (residual !== 0) notableSynergyCandidates.push({ pair: [a, b], residual });
    }
  }
  for (let i = 0; i < redChampions.length; i++) {
    for (let j = i + 1; j < redChampions.length; j++) {
      const a = redChampions[i];
      const b = redChampions[j];
      const residual = synergy.synergy[synergyKey(a, b)]?.residual ?? 0;
      if (residual !== 0) notableSynergyCandidates.push({ pair: [a, b], residual });
    }
  }

  let matchupDiff = 0;
  const notableMatchupCandidates: { pair: [string, string]; residual: number }[] = [];
  for (const role of ROLES) {
    const blueChamp = blue[role];
    const redChamp = red[role];
    if (!blueChamp || !redChamp) continue;
    const residual = synergy.matchup[matchupKey(blueChamp, redChamp)]?.residual ?? 0;
    matchupDiff += residual;
    if (residual !== 0) notableMatchupCandidates.push({ pair: [blueChamp, redChamp], residual });
  }

  // Meta-presence feature: how contested each champion is in pro drafts
  // (pickRate + banRate over the training window). Mirrors the pipeline's
  // train_model.compute_presence_diff exactly; presenceWeight is 0 on model
  // snapshots that don't include the feature.
  const presenceOf = (name: string) => {
    const r = ratings.champions[name];
    return r ? (r.pickRate ?? 0) + (r.banRate ?? 0) : 0;
  };
  const presenceDiff =
    blueChampions.reduce((s, c) => s + presenceOf(c), 0) -
    redChampions.reduce((s, c) => s + presenceOf(c), 0);

  // Team-strength term (0 unless both teams are provided and known —
  // mirrors the pipeline, where an unknown gameid contributes a 0 gap).
  const teamContext = resolveTeamContext(teamOptions);
  const teamEloDiff = teamContext?.eloDiff ?? 0;

  const {
    scoreDiffWeight,
    blueSideBias,
    intercept,
    synergyWeight = 0,
    matchupWeight = 0,
    presenceWeight = 0,
    teamEloWeight = 0,
  } = model.coefficients;
  const logit =
    intercept +
    scoreDiffWeight * (blueTeamScore - redTeamScore) +
    synergyWeight * synergyDiff +
    matchupWeight * matchupDiff +
    presenceWeight * presenceDiff +
    teamEloWeight * teamEloDiff +
    blueSideBias;
  const blueWinProbability = sigmoid(logit);

  return {
    patch: ratings.patch,
    blueWinProbability,
    redWinProbability: 1 - blueWinProbability,
    blueTeamScore,
    redTeamScore,
    synergyDiff,
    matchupDiff,
    breakdown: { blue: blueBreakdown, red: redBreakdown },
    modelMetrics: model.metrics,
    notablePairs: {
      synergy: topNotablePairs(notableSynergyCandidates, 3),
      matchup: topNotablePairs(notableMatchupCandidates, 3),
    },
    teamContext,
  };
}
