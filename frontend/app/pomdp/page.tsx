"use client";

import { useState } from "react";
import ResultBox from "@/components/ResultBox";
import { pomdpBuild, pomdpIntegrated } from "@/lib/api";

export default function POmdpPage() {
  const [dataPath, setDataPath] = useState("");
  const [priorsPath, setPriorsPath] = useState("");
  const [mode, setMode] = useState<"build" | "integrated">("integrated");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    setResult("");
    try {
      if (mode === "build") {
        const r = await pomdpBuild(dataPath, priorsPath);
        setResult(r.result);
      } else {
        const r = await pomdpIntegrated(dataPath);
        setResult(r.result);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">POMDP</h1>
        <p className="text-sm text-gray-500 mt-1">
          Partially-observed Markov decision process for sensor reliability and maintenance.
        </p>
      </div>

      <div className="card space-y-4">
        <div className="flex gap-2">
          {(["integrated", "build"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
                mode === m
                  ? "bg-brand-600 text-white border-brand-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-brand-500"
              }`}
            >
              {m === "integrated" ? "Integrated (recommended)" : "Build only"}
            </button>
          ))}
        </div>

        <div>
          <label className="label">Sensor data path (optional)</label>
          <input
            className="input"
            placeholder="Leave blank to use default"
            value={dataPath}
            onChange={(e) => setDataPath(e.target.value)}
          />
        </div>

        {mode === "build" && (
          <div>
            <label className="label">Priors path (optional)</label>
            <input
              className="input"
              placeholder="Leave blank to use default"
              value={priorsPath}
              onChange={(e) => setPriorsPath(e.target.value)}
            />
          </div>
        )}

        <button
          className="btn-primary"
          onClick={run}
          disabled={loading}
        >
          {loading ? "Running…" : "Run POMDP"}
        </button>
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
