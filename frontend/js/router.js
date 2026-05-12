import { el, qsa } from "./dom.js";
import { state } from "./state.js";

export const pageMeta = {
  timeline: {
    title: "Case Review",
    subtitle:
      "Review prepared videos and inspect how the warning score changes over time while staying synced with playback.",
  },
  videos: {
    title: "Demo Videos",
    subtitle: "Prepared videos exposed by the backend.",
  },
  runs: {
    title: "Technical",
    subtitle: "Experiments, summaries, and comparison studies.",
  },
};

export function setRoute(route) {
  state.route = route;
  const meta = pageMeta[route] || pageMeta.timeline;
  el.pageTitle.textContent = meta.title;
  el.pageSubtitle.textContent = meta.subtitle;

  Object.entries(el.views).forEach(([name, node]) => {
    node.classList.toggle("view--active", name === route);
  });

  qsa(".nav__item", el.navMenu).forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.route === route);
  });
}
