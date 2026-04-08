"use client";

import { useEffect, useState } from "react";
import ResultBox from "@/components/ResultBox";
import { getScenarios, showCausalAnalysis, getDagImageUrl } from "@/lib/api";

export default function CausalPage() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dagTs, setDagTs] = useState(Date.now());

  useEffect(() => {
    getScenarios()
      .then((r) => {
        setScenarios(r.scenarios);
        if (r.scenarios.length > 0) setSelected(r.scenarios[0]);
      })
      .catch(() => {});
  }, []);

  async function runAnalysis() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setResult("");
    try {
      const r = await showCausalAnalysis(selected);
      setResult(r.result);
      setDagTs(Date.now());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Causal Analysis</h1>
        <p className="text-sm text-gray-500 mt-1">
          DAG visualisation, identifiability, and causal effect estimation.
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
              <option value="">Loading…</option>
            )}
            {scenarios.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-2 flex-wrap">
          <button
            className="btn-primary"
            onClick={runAnalysis}
            disabled={loading || !selected}
          >
            {loading ? "Analysing…" : "Run causal analysis"}
          </button>
          <button
            className="btn-secondary"
            onClick={() => setDagTs(Date.now())}
          >
            Refresh DAG image
          </button>
        </div>
      </div>

      {/* DAG image */}
      <div className="card">
        <p className="section-title">Causal DAG</p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${getDagImageUrl()}?t=${dagTs}`}
          alt="Causal DAG"
          className="max-w-full rounded-lg border border-gray-100"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
