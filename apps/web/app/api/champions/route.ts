import { NextResponse } from "next/server";
import { DataNotReadyError, loadChampionRatings } from "@/lib/data";

export async function GET() {
  try {
    const ratings = loadChampionRatings();
    const champions = Object.entries(ratings.champions)
      .map(([name, rating]) => ({ name, ...rating }))
      .sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({
      patch: ratings.patch,
      generatedAt: ratings.generatedAt,
      params: ratings.params,
      champions,
    });
  } catch (err) {
    if (err instanceof DataNotReadyError) {
      return NextResponse.json({ error: err.message }, { status: 503 });
    }
    console.error(err);
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
