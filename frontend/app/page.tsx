"use client";

import { useState } from "react";
import ResultBox from "@/components/ResultBox";
import { unifiedQuery, causalAsk, getScenarios } from "@/lib/api";
import { useEffect } from "react";

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [scenario, setScenario] = useState("");
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [mode, setMode] = useState<"unified" | "causal">("unified");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getScenarios()
      .then((r) => setScenarios(r.scenarios))
      .catch(() => {});
  }, []);

  async function submit() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult("");
    try {
      if (mode === "unified") {
        const r = await unifiedQuery(question);
        setResult(r.result);
      } else {
        const r = await causalAsk(question, scenario);
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
        <h1 className="text-2xl font-bold text-gray-900">Ask a Question</h1>
        <p className="text-sm text-gray-500 mt-1">
          Unified inference chain: RAG retrieval + causal analysis + LLM answer.
        </p>
      </div>

      <div className="card space-y-4">
        {/* Mode */}
        <div className="flex gap-2">
          {(["unified", "causal"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
                mode === m
                  ? "bg-brand-600 text-white border-brand-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-brand-500"
              }`}
            >
              {m === "unified" ? "Unified (recommended)" : "Causal only"}
            </button>
          ))}
        </div>

        {/* Question */}
        <div>
          <label className="label">Question</label>
          <textarea
            className="input resize-none h-24"
            placeholder="e.g. What effect does China's export policy have on graphite prices?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
          />
        </div>

        {/* Scenario (causal mode) */}
        {mode === "causal" && scenarios.length > 0 && (
          <div>
            <label className="label">Scenario (optional)</label>
            <select
              className="input"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            >
              <option value="">— none —</option>
              {scenarios.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={submit}
          disabled={loading || !question.trim()}
        >
          {loading ? "Running…" : "Ask"}
        </button>
      </div>

      <ResultBox content={result} loading={loading} error={error} />
    </div>
  );
}
