"use client";

import { useEffect, useState } from "react";
import ResultBox from "@/components/ResultBox";
import {
  kgRebuild,
  kgShock,
  kgIdentifiability,
  kgDagEdges,
  kgDagInteractive,
  kgEnrich,
  kgBatchEnrich,
  getKgShockSources,
} from "@/lib/api";

type Tab = "dag" | "shock" | "identifiability" | "enrich";

export default function KGPage() {
  const [tab, setTab] = useState<Tab>("dag");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dagHtml, setDagHtml] = useState("");
  const [simplified, setSimplified] = useState(true);

  // Shock
  const [shockSources, setShockSources] = useState<string[]>([]);
  const [shockOrigin, setShockOrigin] = useState("");

  // Enrich
  const [enrichQuery, setEnrichQuery] = useState("");
  const [enrichTopK, setEnrichTopK] = useState(5);

  useEffect(() => {
    getKgShockSources()
      .then((r) => {
        setShockSources(r.sources);
        if (r.sources.length > 0) setShockOrigin(r.sources[0]);
      })
      .catch(() => {});
  }, []);

  async function withLoading(fn: () => Promise<void>) {
    setLoading(true);
    setError(null);
    setResult("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function loadDag() {
    await withLoading(async () => {
      const r = await kgDagInteractive(simplified);
      setDagHtml(r.html);
    });
  }

  async function rebuild() {
    await withLoading(async () => {
      const r = await kgRebuild();
      setResult(r.summary);
      if (r.shock_sources.length > 0) {
        setShockSources(r.shock_sources);
        setShockOrigin(r.shock_sources[0]);
      }
    });
  }

  async function runShock() {
    if (!shockOrigin) return;
    await withLoading(async () => {
      const r = await kgShock(shockOrigin);
      setResult(r.result);
    });
  }

  async function runIdentifiability() {
    await withLoading(async () => {
      const r = await kgIdentifiability();
      setResult(r.result);
    });
  }

  async function runEdges() {
    await withLoading(async () => {
      const r = await kgDagEdges();
      setResult(r.result);
    });
  }

  async function runEnrich() {
    await withLoading(async () => {
      const r = await kgEnrich(enrichQuery, enrichTopK);
      setResult(r.result);
    });
  }

  async function runBatchEnrich() {
    await withLoading(async () => {
      const r = await kgBatchEnrich(enrichTopK);
      setResult(r.result);
    });
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "dag", label: "DAG Viewer" },
    { id: "shock", label: "Shock Propagation" },
    { id: "identifiability", label: "Identifiability" },
    { id: "enrich", label: "Enrich KG" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Knowledge Graph</h1>
        <p className="text-sm text-gray-500 mt-1">
          Explore causal relationships, propagate shocks, and check identifiability.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => {
              setTab(t.id);
              setResult("");
              setError(null);
            }}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              tab === t.id
                ? "border-brand-600 text-brand-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* DAG Viewer */}
      {tab === "dag" && (
        <div className="space-y-4">
          <div className="card space-y-3">
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={simplified}
                  onChange={(e) => setSimplified(e.target.checked)}
                  className="rounded"
                />
                Simplified (top 42 nodes)
              </label>
              <button
                className="btn-primary"
                onClick={loadDag}
                disabled={loading}
              >
                {loading ? "Loading…" : "Render DAG"}
              </button>
              <button
                className="btn-secondary"
                onClick={rebuild}
                disabled={loading}
              >
                Rebuild KG
              </button>
              <button
                className="btn-secondary"
                onClick={runEdges}
                disabled={loading}
              >
                List edges
              </button>
            </div>
          </div>

          {error && (
            <div className="result-box border-red-200 bg-red-50 text-red-700">
              {error}
            </div>
          )}

          {dagHtml && (
            <div
              className="rounded-xl border border-gray-200 overflow-hidden bg-white"
              dangerouslySetInnerHTML={{ __html: dagHtml }}
            />
          )}

          <ResultBox content={result} loading={loading && !dagHtml} />
        </div>
      )}

      {/* Shock Propagation */}
      {tab === "shock" && (
        <div className="space-y-4">
          <div className="card space-y-4">
            <div>
              <label className="label">Shock origin</label>
              {shockSources.length > 0 ? (
                <select
                  className="input"
                  value={shockOrigin}
                  onChange={(e) => setShockOrigin(e.target.value)}
                >
                  {shockSources.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="input"
                  placeholder="e.g. china_export_controls"
                  value={shockOrigin}
                  onChange={(e) => setShockOrigin(e.target.value)}
                />
              )}
            </div>
            <button
              className="btn-primary"
              onClick={runShock}
              disabled={loading || !shockOrigin}
            >
              {loading ? "Propagating…" : "Run shock"}
            </button>
          </div>
          <ResultBox content={result} loading={loading} error={error} />
        </div>
      )}

      {/* Identifiability */}
      {tab === "identifiability" && (
        <div className="space-y-4">
          <div className="card space-y-3">
            <p className="text-sm text-gray-600">
              Checks whether causal effects are identifiable via do-calculus (Pearl) for key
              treatment-outcome pairs in the KG-derived DAG.
            </p>
            <button
              className="btn-primary"
              onClick={runIdentifiability}
              disabled={loading}
            >
              {loading ? "Running…" : "Run identifiability check"}
            </button>
          </div>
          <ResultBox content={result} loading={loading} error={error} />
        </div>
      )}

      {/* Enrich */}
      {tab === "enrich" && (
        <div className="space-y-4">
          <div className="card space-y-4">
            <div>
              <label className="label">Query</label>
              <input
                className="input"
                placeholder="e.g. lithium supply chain Australia"
                value={enrichQuery}
                onChange={(e) => setEnrichQuery(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="label mb-0">Top K</label>
              <input
                type="number"
                className="input w-20"
                min={1}
                max={20}
                value={enrichTopK}
                onChange={(e) => setEnrichTopK(Number(e.target.value))}
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              <button
                className="btn-primary"
                onClick={runEnrich}
                disabled={loading || !enrichQuery.trim()}
              >
                {loading ? "Enriching…" : "Enrich from query"}
              </button>
              <button
                className="btn-secondary"
                onClick={runBatchEnrich}
                disabled={loading}
              >
                Batch enrich (all minerals)
              </button>
            </div>
          </div>
          <ResultBox content={result} loading={loading} error={error} />
        </div>
      )}
    </div>
  );
}
