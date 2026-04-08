"use client";

import { useEffect, useState } from "react";
import ResultBox from "@/components/ResultBox";
import { getScenarios, runScenario, runScenarioCausal, runWorkflow } from "@/lib/api";

export default function ScenariosPage() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [mode, setMode] = useState<"full" | "causal" | "workflow">("full");
  const [result, setResult] = useState("");
  const [runDir, setRunDir] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getScenarios()
      .then((r) => {
        setScenarios(r.scenarios);
        if (r.scenarios.length > 0) setSelected(r.scenarios[0]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  async function run() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setResult("");
    setRunDir("");
    try {
      if (mode === "full") {
        const r = await runScenario(selected);
        setResult(r.output);
        setRunDir(r.run_dir);
      } else if (mode === "causal") {
        const r = await runScenarioCausal(selected);
        setResult(r.result);
      } else {
        const r = await runWorkflow(selected);
        setResult(r.output);
        setRunDir(r.run_dir);
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
        <h1 className="text-2xl font-bold text-gray-900">Run Scenario</h1>
        <p className="text-sm text-gray-500 mt-1">
          Execute a YAML scenario through the causal simulation pipeline.
        </p>
      </div>

      <div className="card space-y-4">
        <div>
          <label className="label">Scenario</label>
          <select
            className="input"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {scenarios.length === 0 && (
              <option value="">Loading scenarios…</option>
            )}
            {scenarios.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="label">Run mode</label>
          <div className="flex gap-2 flex-wrap">
            {(
              [
                { id: "full", label: "Full pipeline" },
                { id: "causal", label: "Causal only" },
                { id: "workflow", label: "Unified workflow" },
              ] as const
            ).map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setMode(id)}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
                  mode === id
                    ? "bg-brand-600 text-white border-brand-600"
                    : "bg-white text-gray-600 border-gray-300 hover:border-brand-500"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <button
          className="btn-primary"
          onClick={run}
          disabled={loading || !selected}
        >
          {loading ? "Running…" : "Run"}
        </button>

        {runDir && (
          <p className="text-xs text-gray-500">
            Run saved to: <code className="bg-gray-100 px-1 rounded">{runDir}</code>
          </p>
        )}
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
