/**
 * Typed API client for the FastAPI backend.
 * All requests go through Next.js rewrites → BACKEND_URL/api/...
 */

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Scenarios ────────────────────────────────────────────────────────────────

export const getScenarios = () =>
  get<{ scenarios: string[] }>("/scenarios");

export const runScenario = (scenario_name: string) =>
  post<{ output: string; run_dir: string }>("/scenario/run", { scenario_name });

export const runScenarioCausal = (scenario_name: string) =>
  post<{ result: string }>("/scenario/run-causal", { scenario_name });

// ── Causal Ask ───────────────────────────────────────────────────────────────

export const causalAsk = (question: string, scenario_name = "", top_k = 5) =>
  post<{ result: string }>("/causal/ask", { question, scenario_name, top_k });

export const unifiedQuery = (question: string, top_k = 5) =>
  post<{ result: string }>("/query/unified", { question, top_k });

// ── Causal / DAG ─────────────────────────────────────────────────────────────

export const showCausalAnalysis = (scenario_name: string) =>
  post<{ result: string }>("/causal/analysis", { scenario_name });

export const getDagImageUrl = () => `${BASE}/causal/dag-image`;

export const refreshDagImage = () =>
  post<void>("/causal/dag-image/refresh", {});

// ── RAG ──────────────────────────────────────────────────────────────────────

export const ragSearch = (
  query: string,
  top_k = 5,
  use_kg_context = false,
  use_classic_search_only = false,
) =>
  post<{ result: string }>("/rag/search", {
    query,
    top_k,
    use_kg_context,
    use_classic_search_only,
  });

export const ragAsk = (query: string, top_k = 5) =>
  post<{ answer: string; episode_id: string }>("/rag/ask", { query, top_k });

export const ragFeedback = (episode_id: string, rating: number) =>
  post<{ result: string }>("/rag/feedback", { episode_id, rating });

export const ragMemoryStats = () =>
  get<{ result: string }>("/rag/memory-stats");

export const reindexRag = () => post<{ result: string }>("/rag/reindex", {});

export const buildHippoRag = () =>
  post<{ result: string }>("/rag/build-hipporag", {});

// ── Knowledge Graph ───────────────────────────────────────────────────────────

export const kgRebuild = () =>
  post<{ summary: string; shock_sources: string[] }>("/kg/rebuild", {});

export const getKgShockSources = () =>
  get<{ sources: string[] }>("/kg/shock-sources");

export const kgShock = (origin_id: string) =>
  post<{ result: string }>("/kg/shock", { origin_id });

export const kgIdentifiability = () =>
  get<{ result: string }>("/kg/identifiability");

export const kgDagEdges = () => get<{ result: string }>("/kg/dag-edges");

export const kgDagInteractive = (simplified = true) =>
  post<{ html: string }>("/kg/dag-interactive", { simplified });

export const kgEnrich = (query: string, top_k = 5) =>
  post<{ result: string }>("/kg/enrich", { query, top_k });

export const kgBatchEnrich = (top_k = 3) =>
  post<{ result: string }>("/kg/batch-enrich", { top_k });

// ── Validate ─────────────────────────────────────────────────────────────────

export const validate = (run_dir: string, year?: number) =>
  post<{ result: string }>("/validate", { run_dir, year });

// ── Synthetic Control ────────────────────────────────────────────────────────

export const runSyntheticControl = () =>
  post<{ result: string }>("/synthetic-control", {});

// ── POMDP ────────────────────────────────────────────────────────────────────

export const pomdpBuild = (data_path = "", priors_path = "") =>
  post<{ result: string }>("/pomdp/build", { data_path, priors_path });

export const pomdpIntegrated = (data_path = "") =>
  post<{ result: string }>("/pomdp/integrated", { data_path });

// ── Three Layers ──────────────────────────────────────────────────────────────

export const threeLayers = (params: {
  run_dir?: string;
  layer?: string;
  treatment?: string;
  outcome?: string;
  cf_year?: string;
  cf_value?: string;
}) =>
  post<{ result: string }>("/three-layers", {
    run_dir: "",
    layer: "2 — Intervention",
    treatment: "ExportPolicy",
    outcome: "Price",
    cf_year: "",
    cf_value: "",
    ...params,
  });

// ── Unified Workflow ──────────────────────────────────────────────────────────

export const runWorkflow = (scenario_name: string) =>
  post<{ output: string; run_dir: string }>("/workflow/run", {
    domain: "Mineral supply chain",
    scenario_name,
  });
