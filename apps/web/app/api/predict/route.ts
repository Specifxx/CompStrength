import { NextRequest, NextResponse } from "next/server";
import { DataNotReadyError, loadChampionRatings, loadModel } from "@/lib/data";
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

  try {
    const ratings = loadChampionRatings();
    const model = loadModel();
    const result = predictMatchup(body.blue, body.red, ratings, model);
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
