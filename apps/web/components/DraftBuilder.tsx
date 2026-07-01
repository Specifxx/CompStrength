"use client";

import { useMemo, useState } from "react";
import { ChampionPicker } from "./ChampionPicker";
import { ResultBreakdown } from "./ResultBreakdown";
import { WinBar } from "./WinBar";
import { ROLES, type ChampionListItem, type DraftTeam, type PredictResponse } from "@/lib/types";

const EMPTY_TEAM: DraftTeam = {
  TOP: null,
  JUNGLE: null,
  MID: null,
  BOTTOM: null,
  SUPPORT: null,
};

export function DraftBuilder({
  champions,
  patch,
}: {
  champions: ChampionListItem[];
  patch: string;
}) {
  const [blue, setBlue] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [red, setRed] = useState<DraftTeam>({ ...EMPTY_TEAM });
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const blueTaken = useMemo(
    () => new Set(Object.values(blue).filter((v): v is string => !!v)),
    [blue],
  );
  const redTaken = useMemo(
    () => new Set(Object.values(red).filter((v): v is string => !!v)),
    [red],
  );
  const allTaken = useMemo(() => new Set([...blueTaken, ...redTaken]), [blueTaken, redTaken]);

  const blueComplete = ROLES.every((r) => blue[r]);
  const redComplete = ROLES.every((r) => red[r]);
  const canPredict = blueComplete && redComplete;

  async function handlePredict() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blue, red }),
      });
      const body = await res.json();
      if (!res.ok) {
        setError(body.error ?? "Prediction failed");
        setResult(null);
      } else {
        setResult(body);
      }
    } catch {
      setError("Network error contacting prediction API");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setBlue({ ...EMPTY_TEAM });
    setRed({ ...EMPTY_TEAM });
    setResult(null);
    setError(null);
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 pb-16 pt-10">
      <header>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Comp<span className="text-sky-400">Strength</span>
        </h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-400">
          Build a blue-side and red-side draft and estimate win probability
          from recent, patch-weighted pro play blended with solo queue
          performance. Currently calibrated on patch{" "}
          <span className="font-mono text-slate-300">{patch}</span>. This is a
          draft-strength estimate, not a guaranteed outcome — team comp is one
          input among many that decide a pro match.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <section className="rounded-lg border border-sky-900/60 bg-sky-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-sky-400">
            Blue Side
          </h2>
          <div className="flex flex-col gap-3">
            {ROLES.map((role) => (
              <ChampionPicker
                key={role}
                role={role}
                side="blue"
                value={blue[role]}
                champions={champions}
                takenElsewhere={allTaken}
                onChange={(champion) => setBlue((prev) => ({ ...prev, [role]: champion }))}
              />
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-rose-400">
            Red Side
          </h2>
          <div className="flex flex-col gap-3">
            {ROLES.map((role) => (
              <ChampionPicker
                key={role}
                role={role}
                side="red"
                value={red[role]}
                champions={champions}
                takenElsewhere={allTaken}
                onChange={(champion) => setRed((prev) => ({ ...prev, [role]: champion }))}
              />
            ))}
          </div>
        </section>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={!canPredict || loading}
          onClick={handlePredict}
          className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          {loading ? "Predicting..." : "Predict Winner"}
        </button>
        <button
          type="button"
          onClick={reset}
          className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800"
        >
          Reset
        </button>
        {!canPredict && (
          <span className="text-xs text-slate-500">
            Fill all 10 roles to run a prediction.
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-amber-800 bg-amber-950/40 px-4 py-3 text-sm text-amber-300">
          {error}
        </div>
      )}

      {result && (
        <section className="flex flex-col gap-6 rounded-lg border border-slate-800 bg-slate-900/40 p-5">
          <WinBar blueWinProbability={result.blueWinProbability} />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <ResultBreakdown side="blue" contributions={result.breakdown.blue} />
            <ResultBreakdown side="red" contributions={result.breakdown.red} />
          </div>
          <p className="text-xs text-slate-500">
            Model log-loss {result.modelMetrics.logLoss.toFixed(3)}, accuracy{" "}
            {(result.modelMetrics.accuracy * 100).toFixed(1)}% vs.{" "}
            {(result.modelMetrics.baselineAccuracy * 100).toFixed(1)}% baseline.
            {result.modelMetrics.note ? ` ${result.modelMetrics.note}` : ""}
          </p>
        </section>
      )}
    </div>
  );
}
