import { el, qsa } from "./dom.js";
import { DEFAULT_API_BASE, resetCaches, state } from "./state.js";
import { refreshCore } from "./api.js";
import { setRoute } from "./router.js";
import { renderDemoPage } from "./pages/demo.js";
import { renderVideosPage } from "./pages/videos.js";
import { renderTechnicalPage } from "./pages/technical.js";

async function renderCurrentView(force = false) {
  setRoute(state.route);
  switch (state.route) {
    case "timeline":
      await renderDemoPage(force);
      break;
    case "videos":
      await renderVideosPage(force);
      break;
    case "runs":
      await renderTechnicalPage(force);
      break;
    default:
      await renderDemoPage(force);
  }
}

function bindUI() {
  el.apiBaseInput.value = state.apiBase;

  qsa(".nav__item", el.navMenu).forEach((btn) => {
    btn.addEventListener("click", async () => {
      state.route = btn.dataset.route;
      await renderCurrentView(false);
    });
  });

  el.reloadAllBtn?.addEventListener("click", async () => {
    resetCaches();
    try {
      await refreshCore();
      await renderCurrentView(true);
    } catch (_) {}
  });

  el.apiBaseInput?.addEventListener("change", async (e) => {
    state.apiBase = e.target.value.trim() || DEFAULT_API_BASE;
    localStorage.setItem("psmokenet_api_base", state.apiBase);
    resetCaches();

    try {
      await refreshCore();
      await renderCurrentView(true);
    } catch (_) {}
  });
}

async function init() {
  bindUI();
  try {
    await refreshCore();
  } catch (_) {}
  await renderCurrentView(false);
}

init();
