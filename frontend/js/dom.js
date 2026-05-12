export function qs(selector, root = document) {
  return root.querySelector(selector);
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export const el = {
  pageTitle: document.getElementById("pageTitle"),
  pageSubtitle: document.getElementById("pageSubtitle"),
  navMenu: document.getElementById("navMenu"),
  apiBaseInput: document.getElementById("apiBaseInput"),
  reloadAllBtn: document.getElementById("reloadAllBtn"),
  backendStatusPill: document.getElementById("backendStatusPill"),
  globalBanner: document.getElementById("globalBanner"),
  views: {
    timeline: document.getElementById("view-timeline"),
    videos: document.getElementById("view-videos"),
    runs: document.getElementById("view-runs"),
  },
};
