"use client";

import { useState } from "react";
import ResultBox from "@/components/ResultBox";
import { validate, runSyntheticControl } from "@/lib/api";

export default function ValidatePage() {
  const [runDir, setRunDir] = useState("");
  const [year, setYear] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runValidate() {
    setLoading(true);
    setError(null);
    setResult("");
    try {
      const r = await validate(runDir, year ? parseInt(year) : undefined);
      setResult(r.result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runSC() {
    setLoading(true);
    setError(null);
    setResult("");
    try {
      const r = await runSyntheticControl();
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
        <h1 className="text-2xl font-bold text-gray-900">Validate</h1>
        <p className="text-sm text-gray-500 mt-1">
          Validate scenario outputs with RAG evidence and synthetic control.
        </p>
      </div>

      <div className="card space-y-4">
        <div>
          <label className="label">Run directory</label>
          <input
            className="input"
            placeholder="e.g. runs/graphite_2008_multishock/20260129_132854"
            value={runDir}
            onChange={(e) => setRunDir(e.target.value)}
          />
        </div>
        <div>
          <label className="label">Year (optional)</label>
          <input
            className="input w-32"
            type="number"
            placeholder="e.g. 2008"
            value={year}
            onChange={(e) => setYear(e.target.value)}
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          <button
            className="btn-primary"
            onClick={runValidate}
            disabled={loading}
          >
            {loading ? "Validating…" : "Validate with RAG"}
          </button>
          <button
            className="btn-secondary"
            onClick={runSC}
            disabled={loading}
          >
            Synthetic control
          </button>
        </div>
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
