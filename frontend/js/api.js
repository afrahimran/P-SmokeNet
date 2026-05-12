import { state } from "./state.js";
import { setBanner, setBackendStatus } from "./ui.js";

async function fetchJson(path, init) {
  const res = await fetch(`${state.apiBase}${path}`, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText || "Request failed");
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchJson("/health"),
  meta: () => fetchJson("/api/meta"),
  listRuns: () => fetchJson("/api/runs"),
  getRun: (runName) => fetchJson(`/api/runs/${encodeURIComponent(runName)}`),
  listExperiments: (runName) =>
    fetchJson(`/api/runs/${encodeURIComponent(runName)}/experiments`),
  experimentSummary: (runName, expName) =>
    fetchJson(`/api/experiments/${encodeURIComponent(runName)}/${encodeURIComponent(expName)}/summary`),
  listVideos: () => fetchJson("/api/videos"),
  demoVideos: () => fetchJson("/api/demo-videos"),
  videoEvents: (videoId, runName) =>
    fetchJson(
      `/api/videos/${encodeURIComponent(videoId)}/events${
        runName ? `?run_name=${encodeURIComponent(runName)}` : ""
      }`
    ),
  inferTimeline: (videoId, runName, expName) => {
    const params = new URLSearchParams();
    if (runName) params.set("run_name", runName);
    if (expName) params.set("exp_name", expName);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return fetchJson(`/api/inference/timeline/${encodeURIComponent(videoId)}${suffix}`);
  },
  listComparisons: () => fetchJson("/api/comparisons"),
  getComparisonFile: (fileName) =>
    fetchJson(`/api/comparisons/${encodeURIComponent(fileName)}`),
  demoVideoUrl: (path) => `${state.apiBase}${path.startsWith("/") ? path : `/static/demo-videos/${path}`}`,
};

export async function refreshCore() {
  try {
    const [health, meta] = await Promise.all([api.health(), api.meta()]);
    state.cache.health = health;
    state.cache.meta = meta;
    if (!state.technical.selectedRun) {
      state.technical.selectedRun = meta.selected_run_name || "";
    }
    setBackendStatus(true, "Backend connected");
    setBanner("");
  } catch (error) {
    setBackendStatus(false, "Backend offline");
    setBanner(`Could not connect to backend at ${state.apiBase}. ${error.message}`, true);
    throw error;
  }
}

export async function ensureRuns(force = false) {
  if (!force && state.cache.runs) return state.cache.runs;
  const runs = await api.listRuns();
  state.cache.runs = runs;
  return runs;
}

export async function ensureRunDetail(runName, force = false) {
  if (!force && state.cache.runDetail.has(runName)) {
    return state.cache.runDetail.get(runName);
  }
  const data = await api.getRun(runName);
  state.cache.runDetail.set(runName, data);
  return data;
}

export async function ensureExperiments(runName, force = false) {
  if (!force && state.cache.experiments.has(runName)) {
    return state.cache.experiments.get(runName);
  }
  const data = await api.listExperiments(runName);
  state.cache.experiments.set(runName, data);
  return data;
}

export async function ensureExperimentSummary(runName, expName, force = false) {
  const key = `${runName}::${expName}`;
  if (!force && state.cache.experimentSummary.has(key)) {
    return state.cache.experimentSummary.get(key);
  }
  const data = await api.experimentSummary(runName, expName);
  state.cache.experimentSummary.set(key, data);
  return data;
}

export async function ensureDemoVideos(force = false) {
  if (!force && state.cache.demoVideos) return state.cache.demoVideos;
  const data = await api.demoVideos();
  state.cache.demoVideos = data;
  return data;
}

export async function ensureTimeline(videoId, runName, expName, force = false) {
  const key = `${videoId}::${runName || ""}::${expName || ""}`;
  if (!force && state.cache.timeline.has(key)) return state.cache.timeline.get(key);
  const data = await api.inferTimeline(videoId, runName, expName);
  state.cache.timeline.set(key, data);
  return data;
}

export async function ensureVideoEvents(videoId, runName, force = false) {
  const key = `${videoId}::${runName || ""}`;
  if (!force && state.cache.videoEvents.has(key)) return state.cache.videoEvents.get(key);
  const data = await api.videoEvents(videoId, runName);
  state.cache.videoEvents.set(key, data);
  return data;
}

export async function ensureComparisons(force = false) {
  if (!force && state.cache.comparisons) return state.cache.comparisons;

  const payload = await api.listComparisons();
  const files = Array.isArray(payload?.files) ? payload.files : [];
  const groups = [];

  for (const entry of files) {
    const fileName = typeof entry === "string" ? entry : entry.file_name;
    const label = typeof entry === "string" ? entry : entry.label || entry.file_name;
    const detail = await api.getComparisonFile(fileName);
    groups.push({
      fileName,
      label,
      path: detail.path || "",
      rows: Array.isArray(detail.rows) ? detail.rows : [],
    });
  }

  state.cache.comparisons = groups;
  return groups;
}
