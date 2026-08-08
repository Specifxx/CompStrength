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
    // Player-level feature weights (mean player-Elo gap / eloScale, and
    // player-champion proficiency sum). Both ride on the optional team
    // selection — 0-features when teams are blank. Absent on older
    // snapshots — treat as `?? 0`.
    playerEloWeight?: number;
    profWeight?: number;
    // Riot Global Power Rankings gap (gprScale-divided) and early-game
    // gold-at-15 rating gap (econScale-divided). Both ride on the optional
    // team selection. Absent on older snapshots — treat as `?? 0`.
    gprWeight?: number;
    econWeight?: number;
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
  // Same held-out predictions, restricted to the newest season's patches
  // (e.g. 16.x): the number that matches predicting current games with all
  // history available. Null/absent on single-season reports.
  currentSeasonMetrics?: {
    testGames: number;
    patchMajor: number;
    accuracy: number;
    logLoss: number;
    baselineAccuracy: number;
    draftOnlyAccuracy: number;
    draftOnlyLogLoss: number;
  } | null;
  teamFeatureUsed?: boolean;
  // Player Elo + champion proficiency (players.py) — ride on the same
  // optional team selection. Absent on older reports.
  playerFeaturesUsed?: boolean;
  // Riot GPR gap and early-game (gold-at-15) rating gap. Absent on older
  // reports.
  gprFeatureUsed?: boolean;
  econFeatureUsed?: boolean;
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
  // Optional per-side player-name overrides (5 entries, ROLES order). Absent
  // -> the team's inferred roster is used for the player features.
  bluePlayers?: (string | null)[] | null;
  redPlayers?: (string | null)[] | null;
}

/** One seat of a team's current roster: the player and their position
 *  (Oracle's Elixir codes: top/jng/mid/bot/sup). Per-player stats live in
 *  the separate players index (see {@link PlayersFile}). */
export interface RosterSeat {
  player: string;
  position: string;
}

/** One professional team's Elo rating snapshot (data/teams.json). */
export interface TeamRating {
  elo: number;
  games: number;
  league: string;
  lastPlayed: string;
  // Opponent-adjusted expected gold lead at 15 minutes, in gold (0 = exactly
  // average). Absent on older snapshots or when the feature is disabled.
  econ?: number;
  // Riot's published Global Power Ranking score. Present only for the ~58
  // tier-1 orgs Riot rates; absent for everyone else, which zeroes the GPR
  // term for that matchup.
  gpr?: number;
  // The five (player, position) seats from the team's most recent game.
  // Absent on older snapshots or when player features are disabled.
  roster?: RosterSeat[];
}

export interface TeamsFile {
  generatedAt: string;
  patch: string;
  params: {
    eloK: number;
    eloScale: number;
    initialElo: number;
    // Divisors for the early-game and GPR gaps. Absent on older snapshots.
    econScale?: number;
    gprScale?: number;
    // Pseudo-games shrinking a player-champion winrate toward the player's
    // own overall winrate (proficiency). Absent on older snapshots.
    profShrink?: number;
  };
  teams: Record<string, TeamRating>;
}

/** One player's "as of today" stats (data/players.json): sequential pre-game
 *  Elo, overall record, and per-champion record ({champion: [wins, games]}) —
 *  everything needed to reproduce the pipeline's player features client-side
 *  for any player, including a substitute the user edits in. */
export interface PlayerStat {
  elo: number;
  wins: number;
  games: number;
  champions: Record<string, [number, number]>;
}

export interface PlayersFile {
  generatedAt: string;
  patch: string;
  params: {
    // Pseudo-games shrinking a player-champion winrate toward the player's
    // own overall winrate (proficiency). Absent on older snapshots.
    profShrink?: number;
  };
  players: Record<string, PlayerStat>;
}

/** Team-strength inputs actually applied to a prediction. */
export interface TeamContext {
  blueTeam: string;
  redTeam: string;
  blueElo: number;
  redElo: number;
  /** (blueElo - redElo) / eloScale — the model feature value. */
  eloDiff: number;
  // Leagues of the two teams (from teams.json). When they differ, this is a
  // cross-region matchup (e.g. an international event like MSI): team Elo is
  // calibrated mostly WITHIN each league and only weakly across regions, so
  // the prediction is measurably less reliable (~55% held-out on
  // international events vs ~65% same-region). The UI surfaces a caveat.
  blueLeague?: string;
  redLeague?: string;
  crossRegion?: boolean;
  // Player-level add-on (present when BOTH teams ship a roster): the
  // inferred starting fives, their mean Elos, and the two feature values
  // actually fed to the model. Absent when either roster is missing (the
  // player terms are then 0 — "unknown players").
  blueRoster?: string[];
  redRoster?: string[];
  bluePlayerElo?: number;
  redPlayerElo?: number;
  /** (mean blue player Elo - mean red) / eloScale — the model feature. */
  playerEloDiff?: number;
  /** Blue proficiency sum - red proficiency sum on the drafted champions. */
  profDiff?: number;
  // Early-game (gold-at-15) team ratings and the model feature they produce.
  // Present when both teams carry an `econ` rating.
  blueEcon?: number;
  redEcon?: number;
  /** (blueEcon - redEcon) / econScale — the model feature value. */
  econDiff?: number;
  // Riot Global Power Rankings, present only when BOTH teams are tier-1 orgs
  // Riot rates. Absent otherwise, and the GPR term is then 0.
  blueGpr?: number;
  redGpr?: number;
  /** (blueGpr - redGpr) / gprScale — the model feature value. */
  gprDiff?: number;
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
