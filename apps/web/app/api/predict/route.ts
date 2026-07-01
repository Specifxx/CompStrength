import { NextRequest, NextResponse } from "next/server";
import { DataNotReadyError, loadChampionRatings, loadModel, loadSynergy, loadTeams } from "@/lib/data";
import {
  IncompleteDraftError,
  UnknownChampionError,
  predictMatchup,
} from "@/lib/predict";
import { ROLES, type DraftTeam, type PredictRequest } from "@/lib/types";

function isDraftTeam(value: unknown): value is DraftTeam {
  if (typeof value !== "object" || value === null) return false;
  const team = value as Record<string, unknown>;
  return ROLES.every((role) => {
    const v = team[role];
    return v === null || v === undefined || typeof v === "string";
  });
}

export async function POST(request: NextRequest) {
  let body: PredictRequest;
  try {
    body = (await request.json()) as PredictRequest;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!isDraftTeam(body?.blue) || !isDraftTeam(body?.red)) {
    return NextResponse.json(
      { error: "Body must include `blue` and `red` team objects keyed by role" },
      { status: 400 },
    );
  }

  const blueTeam = typeof body.blueTeam === "string" ? body.blueTeam : null;
  const redTeam = typeof body.redTeam === "string" ? body.redTeam : null;

  try {
    const ratings = loadChampionRatings();
    const model = loadModel();
    const synergy = loadSynergy();
    // teams.json is optional (older snapshots) -- degrade to draft-only.
    let teams = null;
    try {
      teams = loadTeams();
    } catch (err) {
      if (!(err instanceof DataNotReadyError)) throw err;
    }
    const result = predictMatchup(body.blue, body.red, ratings, model, synergy, {
      blueTeam,
      redTeam,
      teams,
    });
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof DataNotReadyError) {
      return NextResponse.json({ error: err.message }, { status: 503 });
    }
    if (err instanceof IncompleteDraftError || err instanceof UnknownChampionError) {
      return NextResponse.json({ error: err.message }, { status: 400 });
    }
    console.error(err);
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
