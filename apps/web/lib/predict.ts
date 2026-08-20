import {
  ROLES,
  type ChampionContribution,
  type ChampionRatingsFile,
  type DraftTeam,
  type ModelFile,
  type NotablePair,
  type PlayerStat,
  type PlayersFile,
  type PredictResponse,
  type Role,
  type SynergyFile,
  type TeamContext,
  type TeamsFile,
} from "./types";

/** Sequential-Elo starting rating (must match teams.INITIAL_ELO in the
 *  pipeline): an unknown player is treated as a perfectly average one. */
const INITIAL_ELO = 1500;

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
  /** Global per-player stats index (data/players.json). */
  players?: PlayersFile | null;
  /** Per-side player-name overrides in ROLES order (TOP..SUPPORT). When a
   *  side's array is absent, that team's inferred roster is used; when a slot
   *  is null/empty the player is unknown (treated as perfectly average). */
  bluePlayers?: (string | null)[] | null;
  redPlayers?: (string | null)[] | null;
}

/** Draft role -> Oracle's Elixir position code used in roster seats. */
const POSITION_BY_ROLE: Record<Role, string> = {
  TOP: "top",
  JUNGLE: "jng",
  MID: "mid",
  BOTTOM: "bot",
  SUPPORT: "sup",
};

/** Mirror of the pipeline's players.proficiency(): the player's shrunk
 *  winrate edge on ``champion`` over their own overall winrate. 0 for
 *  unknown players/champions. */
function proficiency(stat: PlayerStat | undefined, champion: string, shrink: number): number {
  if (!stat) return 0;
  const base = stat.games > 0 ? stat.wins / stat.games : 0.5;
  const [w, g] = stat.champions[champion] ?? [0, 0];
  return (w + shrink * base) / (g + shrink) - base;
}

/** The five player names for one side, in ROLES order. Overrides win; else
 *  the team's inferred roster (mapped by position) fills each role. Returns
 *  null if fewer than 5 names can be resolved (player features then stay
 *  off for that prediction). */
function playersByRole(
  teamRoster: { player: string; position: string }[] | undefined,
  overrides: (string | null)[] | null | undefined,
): (string | null)[] | null {
  const names = ROLES.map((role, i) => {
    const override = overrides?.[i];
    if (override && override.trim()) return override.trim();
    const seat = teamRoster?.find((s) => s.position === POSITION_BY_ROLE[role]);
    return seat?.player ?? null;
  });
  return names.every((n) => n) ? names : null;
}

/** Resolve the team-strength (Elo gap) term, plus the player-level add-on
 *  (mean player Elo gap + champion proficiency) when both sides have a full
 *  five. Both-or-nothing at every level: the team term only applies when BOTH
 *  org names are given and found in teams.json — a single known team vs an
 *  unspecified opponent would silently be treated as "vs a perfectly average
 *  team", which is misleading rather than neutral. Likewise the player terms
 *  only apply when BOTH sides resolve five players. */
function resolveTeamContext(
  options: TeamOptions | undefined,
  blue: DraftTeam,
  red: DraftTeam,
): TeamContext | null {
  if (!options?.teams || !options.blueTeam || !options.redTeam) return null;
  const blueRating = options.teams.teams[options.blueTeam];
  const redRating = options.teams.teams[options.redTeam];
  if (!blueRating || !redRating) return null;
  const scale = options.teams.params.eloScale || 400;
  const blueLeague = blueRating.league || "";
  const redLeague = redRating.league || "";
  const context: TeamContext = {
    blueTeam: options.blueTeam,
    redTeam: options.redTeam,
    blueElo: blueRating.elo,
    redElo: redRating.elo,
    eloDiff: (blueRating.elo - redRating.elo) / scale,
    blueLeague,
    redLeague,
    // Cross-region only when both leagues are known and differ (an
    // international-style matchup). Elo bridges regions only through the few
    // inter-region games, so these predictions carry more uncertainty.
    crossRegion: !!blueLeague && !!redLeague && blueLeague !== redLeague,
  };

  const index = options.players?.players;
  const blueNames = playersByRole(blueRating.roster, options.bluePlayers);
  const redNames = playersByRole(redRating.roster, options.redPlayers);
  if (index && blueNames && redNames) {
    const shrink =
      options.teams.params.profShrink ?? options.players?.params.profShrink ?? 8;
    const eloOf = (name: string) => index[name]?.elo ?? INITIAL_ELO;
    const mean = (names: (string | null)[]) =>
      names.reduce((s, n) => s + eloOf(n as string), 0) / names.length;
    const sideProf = (names: (string | null)[], draft: DraftTeam) =>
      ROLES.reduce((sum, role, i) => {
        const champion = draft[role];
        const name = names[i];
        if (!champion || !name) return sum;
        return sum + proficiency(index[name], champion, shrink);
      }, 0);

    context.blueRoster = blueNames as string[];
    context.redRoster = redNames as string[];
    context.bluePlayerElo = mean(blueNames);
    context.redPlayerElo = mean(redNames);
    context.playerEloDiff = (context.bluePlayerElo - context.redPlayerElo) / scale;
    context.profDiff = sideProf(blueNames, blue) - sideProf(redNames, red);
  }
  return context;
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
  // The player-level terms ride on the same selection: rosters are inferred
  // from the chosen teams' most recent lineup, so no extra inputs needed.
  const teamContext = resolveTeamContext(teamOptions, blue, red);
  const teamEloDiff = teamContext?.eloDiff ?? 0;
  const playerEloDiff = teamContext?.playerEloDiff ?? 0;
  const profDiff = teamContext?.profDiff ?? 0;

  // With no team context, use the model's DEDICATED draft-only fit when the
  // snapshot carries one: reusing the full coefficients with the team terms
  // zeroed is measurably worse (they were fit WITH team features present,
  // and without them the in-sample-fit synergy residual is over-trusted --
  // its weight is 0 in the draft-only block). Older model.json snapshots
  // without the block keep the previous behaviour.
  const activeCoefficients =
    !teamContext && model.draftOnlyCoefficients
      ? model.draftOnlyCoefficients
      : model.coefficients;
  const {
    scoreDiffWeight,
    blueSideBias,
    intercept,
    synergyWeight = 0,
    matchupWeight = 0,
    presenceWeight = 0,
    teamEloWeight = 0,
    playerEloWeight = 0,
    profWeight = 0,
  } = activeCoefficients;
  const logit =
    intercept +
    scoreDiffWeight * (blueTeamScore - redTeamScore) +
    synergyWeight * synergyDiff +
    matchupWeight * matchupDiff +
    presenceWeight * presenceDiff +
    teamEloWeight * teamEloDiff +
    playerEloWeight * playerEloDiff +
    profWeight * profDiff +
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

/** Blue-side win probability tolerant of an INCOMPLETE draft: unfilled roles
 *  and unknown champions contribute 0 to every champion-derived term (the
 *  model's logit is additive, so a missing pick is just a missing summand).
 *  Team/player terms come from the optional team selection as usual. Used to
 *  score candidate picks for the dynamic best-fit suggestions. */
function winProbabilityPartial(
  blue: DraftTeam,
  red: DraftTeam,
  ratings: ChampionRatingsFile,
  model: ModelFile,
  synergy: SynergyFile,
  teamOptions: TeamOptions | undefined,
): number {
  const bFilled = ROLES.map((r) => blue[r]).filter((c): c is string => !!c);
  const rFilled = ROLES.map((r) => red[r]).filter((c): c is string => !!c);
  const strengthOf = (c: string) => ratings.champions[c]?.strengthScore ?? 0;
  const scoreDiff =
    bFilled.reduce((s, c) => s + strengthOf(c), 0) -
    rFilled.reduce((s, c) => s + strengthOf(c), 0);

  const sideSyn = (champs: string[]) => {
    let t = 0;
    for (let i = 0; i < champs.length; i++)
      for (let j = i + 1; j < champs.length; j++)
        t += synergy.synergy[synergyKey(champs[i], champs[j])]?.residual ?? 0;
    return t;
  };
  const synergyDiff = sideSyn(bFilled) - sideSyn(rFilled);

  let matchupDiff = 0;
  for (const role of ROLES) {
    const b = blue[role];
    const r = red[role];
    if (b && r) matchupDiff += synergy.matchup[matchupKey(b, r)]?.residual ?? 0;
  }

  const presenceOf = (c: string) => {
    const x = ratings.champions[c];
    return x ? (x.pickRate ?? 0) + (x.banRate ?? 0) : 0;
  };
  const presenceDiff =
    bFilled.reduce((s, c) => s + presenceOf(c), 0) -
    rFilled.reduce((s, c) => s + presenceOf(c), 0);

  const ctx = resolveTeamContext(teamOptions, blue, red);
  // DELIBERATELY the full coefficient set, not draftOnlyCoefficients: this
  // function only powers suggestPicks, whose "best synergy/counter" pick is
  // ranked by how much the synergy/matchup tables move the probability. The
  // draft-only block zeroes synergyWeight (it hurts calibrated accuracy),
  // which would silence that ranking entirely; as a relative comparison
  // among candidates it doesn't need the calibrated draft-only fit.
  const {
    scoreDiffWeight,
    blueSideBias,
    intercept,
    synergyWeight = 0,
    matchupWeight = 0,
    presenceWeight = 0,
    teamEloWeight = 0,
    playerEloWeight = 0,
    profWeight = 0,
  } = model.coefficients;
  const logit =
    intercept +
    scoreDiffWeight * scoreDiff +
    synergyWeight * synergyDiff +
    matchupWeight * matchupDiff +
    presenceWeight * presenceDiff +
    teamEloWeight * (ctx?.eloDiff ?? 0) +
    playerEloWeight * (ctx?.playerEloDiff ?? 0) +
    profWeight * (ctx?.profDiff ?? 0) +
    blueSideBias;
  return sigmoid(logit);
}

/** A recommended champion for one empty slot and the resulting win
 *  probability FOR THAT SIDE if picked. */
export interface PickSuggestion {
  champion: string;
  sideWinProbability: number;
}

/** Two recommendations for one empty slot: the highest-win-rate champion
 *  (model-faithful, strength-dominated) and — only when it's a DIFFERENT
 *  champion — the one that best fits the comp via synergy with your picks and
 *  counter vs the enemy laner. ``fit`` is omitted when it equals ``best``
 *  (e.g. early in a draft with no context to differentiate on). */
export interface SlotSuggestion {
  best: PickSuggestion;
  fit?: PickSuggestion;
}

/** For every EMPTY role on each side, the available champion that maximises
 *  that side's win probability given the current (partial) draft and the
 *  optional team selection. ROLE-AWARE: each slot only considers champions
 *  whose primary role matches it (otherwise the model's role-agnostic score
 *  would just recommend the single strongest champion for every slot).
 *  Accounts for the opponent's current picks (counter-picking) and synergy
 *  with the side's own picks. Champions already used anywhere are excluded. */
export function suggestPicks(
  blue: DraftTeam,
  red: DraftTeam,
  ratings: ChampionRatingsFile,
  model: ModelFile,
  synergy: SynergyFile,
  champions: { name: string; primaryRole: Role }[],
  teamOptions?: TeamOptions,
): { blue: Partial<Record<Role, SlotSuggestion>>; red: Partial<Record<Role, SlotSuggestion>> } {
  const taken = new Set<string>();
  for (const r of ROLES) {
    if (blue[r]) taken.add(blue[r] as string);
    if (red[r]) taken.add(red[r] as string);
  }
  // Group candidate champions by their primary role, so a TOP slot only ever
  // suggests top laners, a JUNGLE slot junglers, etc.
  const byRole: Record<Role, string[]> = {
    TOP: [], JUNGLE: [], MID: [], BOTTOM: [], SUPPORT: [],
  };
  for (const c of champions) {
    if (ratings.champions[c.name] && byRole[c.primaryRole]) {
      byRole[c.primaryRole].push(c.name);
    }
  }
  // Strength-only synergy view (no pair/matchup residuals) to isolate the
  // comp-dependent part of a candidate's win probability.
  const noSynergy: SynergyFile = { ...synergy, synergy: {}, matchup: {} };

  const forSide = (side: "blue" | "red"): Partial<Record<Role, SlotSuggestion>> => {
    const base = side === "blue" ? blue : red;
    const out: Partial<Record<Role, SlotSuggestion>> = {};
    for (const role of ROLES) {
      if (base[role]) continue; // only suggest for empty slots
      let best: PickSuggestion | null = null;
      // "fit": champion whose synergy-with-picks + counter-vs-enemy moves the
      // win probability the MOST (i.e. the comp-dependent contribution),
      // regardless of raw strength. Tie-broken by overall win probability.
      let fit: { champion: string; sideWinProbability: number; compDelta: number } | null = null;
      for (const champ of byRole[role]) {
        if (taken.has(champ)) continue;
        const b = side === "blue" ? { ...blue, [role]: champ } : blue;
        const r = side === "red" ? { ...red, [role]: champ } : red;
        const blueWin = winProbabilityPartial(b, r, ratings, model, synergy, teamOptions);
        const sideWin = side === "blue" ? blueWin : 1 - blueWin;
        const blueWinNoSyn = winProbabilityPartial(b, r, ratings, model, noSynergy, teamOptions);
        const sideWinNoSyn = side === "blue" ? blueWinNoSyn : 1 - blueWinNoSyn;
        const compDelta = sideWin - sideWinNoSyn; // synergy+counter contribution

        if (!best || sideWin > best.sideWinProbability) {
          best = { champion: champ, sideWinProbability: sideWin };
        }
        if (
          !fit ||
          compDelta > fit.compDelta + 1e-9 ||
          (Math.abs(compDelta - fit.compDelta) <= 1e-9 && sideWin > fit.sideWinProbability)
        ) {
          fit = { champion: champ, sideWinProbability: sideWin, compDelta };
        }
      }
      if (best) {
        const slot: SlotSuggestion = { best };
        // Only surface a distinct synergy pick when it differs from the
        // win-rate pick AND actually gains something from the comp.
        if (fit && fit.champion !== best.champion && fit.compDelta > 1e-4) {
          slot.fit = { champion: fit.champion, sideWinProbability: fit.sideWinProbability };
        }
        out[role] = slot;
      }
    }
    return out;
  };

  return { blue: forSide("blue"), red: forSide("red") };
}
