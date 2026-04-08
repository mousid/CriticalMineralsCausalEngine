"use client";

import { useState } from "react";
import ResultBox from "@/components/ResultBox";
import {
  ragSearch,
  ragAsk,
  ragFeedback,
  ragMemoryStats,
  reindexRag,
  buildHippoRag,
} from "@/lib/api";

type Tab = "search" | "ask" | "manage";

export default function RAGPage() {
  const [tab, setTab] = useState<Tab>("ask");

  // Ask tab
  const [askQuery, setAskQuery] = useState("");
  const [askResult, setAskResult] = useState("");
  const [episodeId, setEpisodeId] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [feedbackMsg, setFeedbackMsg] = useState("");

  // Search tab
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTopK, setSearchTopK] = useState(5);
  const [useKG, setUseKG] = useState(false);
  const [classicOnly, setClassicOnly] = useState(false);
  const [searchResult, setSearchResult] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Manage tab
  const [manageResult, setManageResult] = useState("");
  const [manageLoading, setManageLoading] = useState(false);

  async function submitAsk() {
    if (!askQuery.trim()) return;
    setAskLoading(true);
    setAskError(null);
    setAskResult("");
    setEpisodeId("");
    setFeedbackMsg("");
    try {
      const r = await ragAsk(askQuery);
      setAskResult(r.answer);
      setEpisodeId(r.episode_id);
    } catch (e) {
      setAskError(String(e));
    } finally {
      setAskLoading(false);
    }
  }

  async function submitSearch() {
    if (!searchQuery.trim()) return;
    setSearchLoading(true);
    setSearchError(null);
    setSearchResult("");
    try {
      const r = await ragSearch(searchQuery, searchTopK, useKG, classicOnly);
      setSearchResult(r.result);
    } catch (e) {
      setSearchError(String(e));
    } finally {
      setSearchLoading(false);
    }
  }

  async function sendFeedback(rating: number) {
    if (!episodeId) return;
    try {
      const r = await ragFeedback(episodeId, rating);
      setFeedbackMsg(r.result);
    } catch (e) {
      setFeedbackMsg(String(e));
    }
  }

  async function manageAction(action: "stats" | "reindex" | "hipporag") {
    setManageLoading(true);
    setManageResult("");
    try {
      if (action === "stats") {
        const r = await ragMemoryStats();
        setManageResult(r.result);
      } else if (action === "reindex") {
        const r = await reindexRag();
        setManageResult(r.result);
      } else {
        const r = await buildHippoRag();
        setManageResult(r.result);
      }
    } catch (e) {
      setManageResult(String(e));
    } finally {
      setManageLoading(false);
    }
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "ask", label: "Ask" },
    { id: "search", label: "Search Documents" },
    { id: "manage", label: "Manage Index" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Documents / RAG</h1>
        <p className="text-sm text-gray-500 mt-1">
          Retrieval-augmented Q&A over indexed documents.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
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

      {/* Ask */}
      {tab === "ask" && (
        <div className="space-y-4">
          <div className="card space-y-4">
            <div>
              <label className="label">Question</label>
              <textarea
                className="input resize-none h-24"
                placeholder="e.g. What caused the 2008 graphite trade shock?"
                value={askQuery}
                onChange={(e) => setAskQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey))
                    submitAsk();
                }}
              />
            </div>
            <button
              className="btn-primary"
              onClick={submitAsk}
              disabled={askLoading || !askQuery.trim()}
            >
              {askLoading ? "Thinking…" : "Ask"}
            </button>
          </div>

          <ResultBox
            content={askResult}
            loading={askLoading}
            error={askError}
          />

          {episodeId && !askLoading && (
            <div className="flex items-center gap-3 text-sm text-gray-500">
              <span>Was this helpful?</span>
              <button
                className="btn-secondary py-1 px-3"
                onClick={() => sendFeedback(1)}
              >
                👍
              </button>
              <button
                className="btn-secondary py-1 px-3"
                onClick={() => sendFeedback(-1)}
              >
                👎
              </button>
              {feedbackMsg && (
                <span className="text-gray-400">{feedbackMsg}</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Search */}
      {tab === "search" && (
        <div className="space-y-4">
          <div className="card space-y-4">
            <div>
              <label className="label">Query</label>
              <input
                className="input"
                placeholder="e.g. graphite supply chain China"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submitSearch();
                }}
              />
            </div>
            <div className="flex gap-4 items-center flex-wrap">
              <div className="flex items-center gap-2">
                <label className="label mb-0 whitespace-nowrap">Top K</label>
                <input
                  type="number"
                  className="input w-20"
                  min={1}
                  max={20}
                  value={searchTopK}
                  onChange={(e) => setSearchTopK(Number(e.target.value))}
                />
              </div>
              <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useKG}
                  onChange={(e) => setUseKG(e.target.checked)}
                  className="rounded"
                />
                Include KG context
              </label>
              <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={classicOnly}
                  onChange={(e) => setClassicOnly(e.target.checked)}
                  className="rounded"
                />
                Classic search only
              </label>
            </div>
            <button
              className="btn-primary"
              onClick={submitSearch}
              disabled={searchLoading || !searchQuery.trim()}
            >
              {searchLoading ? "Searching…" : "Search"}
            </button>
          </div>

          <ResultBox
            content={searchResult}
            loading={searchLoading}
            error={searchError}
          />
        </div>
      )}

      {/* Manage */}
      {tab === "manage" && (
        <div className="space-y-4">
          <div className="card space-y-3">
            <p className="text-sm text-gray-600">
              Manage the document index and memory.
            </p>
            <div className="flex gap-2 flex-wrap">
              <button
                className="btn-secondary"
                onClick={() => manageAction("stats")}
                disabled={manageLoading}
              >
                Memory stats
              </button>
              <button
                className="btn-secondary"
                onClick={() => manageAction("reindex")}
                disabled={manageLoading}
              >
                Rebuild index
              </button>
              <button
                className="btn-secondary"
                onClick={() => manageAction("hipporag")}
                disabled={manageLoading}
              >
                Build HippoRAG index
              </button>
            </div>
          </div>

          <ResultBox content={manageResult} loading={manageLoading} />
        </div>
      )}
    </div>
  );
}
