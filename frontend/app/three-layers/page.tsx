"use client";

import { useEffect, useState } from "react";
import ResultBox from "@/components/ResultBox";
import { threeLayers, getScenarios, runScenario } from "@/lib/api";

const LAYERS = [
  "1 — Association",
  "2 — Intervention",
  "3 — Counterfactual",
];

export default function ThreeLayersPage() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [runDir, setRunDir] = useState("");
  const [layer, setLayer] = useState(LAYERS[1]);
  const [treatment, setTreatment] = useState("ExportPolicy");
  const [outcome, setOutcome] = useState("Price");
  const [cfYear, setCfYear] = useState("");
  const [cfValue, setCfValue] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState("");

  useEffect(() => {
    getScenarios()
      .then((r) => {
        setScenarios(r.scenarios);
        if (r.scenarios.length > 0) setSelectedScenario(r.scenarios[0]);
      })
      .catch(() => {});
  }, []);

  async function quickRun() {
    if (!selectedScenario) return;
    setRunLoading(true);
    try {
      const r = await runScenario(selectedScenario);
      setRunDir(r.run_dir);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunLoading(false);
    }
  }

  async function query() {
    setLoading(true);
    setError(null);
    setResult("");
    try {
      const r = await threeLayers({
        run_dir: runDir,
        layer,
        treatment,
        outcome,
        cf_year: cfYear,
        cf_value: cfValue,
      });
      setResult(r.result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Three Layers of Causation</h1>
        <p className="text-sm text-gray-500 mt-1">
          Pearl's causal hierarchy: association, intervention, and counterfactual queries.
        </p>
      </div>

      {/* Quick run */}
      <div className="card space-y-3">
        <p className="text-sm font-medium text-gray-700">
          1. Run a scenario to get a run directory
        </p>
        <div className="flex gap-2 items-center flex-wrap">
          <select
            className="input flex-1 min-w-0"
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
          >
            {scenarios.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button
            className="btn-secondary shrink-0"
            onClick={quickRun}
            disabled={runLoading || !selectedScenario}
          >
            {runLoading ? "Running…" : "Run scenario"}
          </button>
        </div>
        {runDir && (
          <p className="text-xs text-green-600">
            Run dir: <code className="bg-gray-100 px-1 rounded">{runDir}</code>
          </p>
        )}
      </div>

      {/* Query form */}
      <div className="card space-y-4">
        <p className="text-sm font-medium text-gray-700">2. Query the causal model</p>

        <div>
          <label className="label">Run directory</label>
          <input
            className="input"
            placeholder="Populated automatically after running above, or enter manually"
            value={runDir}
            onChange={(e) => setRunDir(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Layer</label>
            <select
              className="input"
              value={layer}
              onChange={(e) => setLayer(e.target.value)}
            >
              {LAYERS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Treatment variable</label>
            <input
              className="input"
              value={treatment}
              onChange={(e) => setTreatment(e.target.value)}
            />
          </div>
          <div>
            <label className="label">Outcome variable</label>
            <input
              className="input"
              value={outcome}
              onChange={(e) => setOutcome(e.target.value)}
            />
          </div>
        </div>

        {layer === LAYERS[2] && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Counterfactual year</label>
              <input
                className="input"
                placeholder="e.g. 2008"
                value={cfYear}
                onChange={(e) => setCfYear(e.target.value)}
              />
            </div>
            <div>
              <label className="label">Counterfactual value</label>
              <input
                className="input"
                placeholder="e.g. 0.5"
                value={cfValue}
                onChange={(e) => setCfValue(e.target.value)}
              />
            </div>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={query}
          disabled={loading}
        >
          {loading ? "Querying…" : "Query"}
        </button>
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
