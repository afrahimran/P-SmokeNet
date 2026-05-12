import { el, qsa } from "../dom.js";
import { escapeHtml, loadingCard } from "../format.js";
import { api, ensureDemoVideos } from "../api.js";
import { attachDemoTimeCap } from "../chart.js";

export async function renderVideosPage(force = false) {
  const root = el.views.videos;
  root.innerHTML = loadingCard("Loading videos…");

  try {
    const demos = await ensureDemoVideos(force);
    root.innerHTML = `
      <div class="grid">
        <section class="card">
          <div class="card__header"><h2 class="card__title">Demo videos</h2></div>
          <div class="card__body">
            <div class="video-grid">
              ${demos
                .map(
                  (v) => `
                  <article class="card video-card">
                    <div class="card__header">
                      <div>
                        <h3 class="card__title">${escapeHtml(v.video_id)}</h3>
                        <p class="card__subtitle">${escapeHtml(v.filename)}</p>
                      </div>
                    </div>
                    <div class="card__body">
                      <video controls preload="metadata">
                        <source src="${escapeHtml(api.demoVideoUrl(v.url))}">
                      </video>
                    </div>
                  </article>
                `
                )
                .join("")}
            </div>
          </div>
        </section>
      </div>
    `;

    qsa("video", root).forEach((videoEl) => attachDemoTimeCap(videoEl));
  } catch (error) {
    root.innerHTML = `<div class="card"><div class="card__body"><p class="muted">Could not load demo videos: ${escapeHtml(error.message)}</p></div></div>`;
  }
}
