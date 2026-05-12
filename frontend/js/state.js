export const DEFAULT_API_BASE =
  localStorage.getItem("psmokenet_api_base") || "http://p-smokenet.onrender.com";

export const DEMO_MAX_SECONDS = 60;

export const state = {
  apiBase: DEFAULT_API_BASE,
  route: "timeline",
  cache: {
    health: null,
    meta: null,
    runs: null,
    demoVideos: null,
    videos: null,
    runDetail: new Map(),
    experiments: new Map(),
    experimentSummary: new Map(),
    comparisons: null,
    timeline: new Map(),
    videoEvents: new Map(),
  },
  timeline: {
    selectedVideo: "",
    videoEl: null,
    currentTime: 0,
    currentProb: 0,
    isAlert: false,
  },
  technical: {
    selectedRun: "",
    selectedExperiment: "",
  },
};

export function resetCaches() {
  state.cache.health = null;
  state.cache.meta = null;
  state.cache.runs = null;
  state.cache.demoVideos = null;
  state.cache.videos = null;
  state.cache.comparisons = null;
  state.cache.runDetail.clear();
  state.cache.experiments.clear();
  state.cache.experimentSummary.clear();
  state.cache.timeline.clear();
  state.cache.videoEvents.clear();
}
