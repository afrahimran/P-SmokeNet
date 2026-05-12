import { el, qs } from "../dom.js";
import { state, DEMO_MAX_SECONDS } from "../state.js";
import { escapeHtml, formatNumber, loadingCard } from "../format.js";
import { api, ensureDemoVideos, ensureTimeline, ensureVideoEvents } from "../api.js";
import {
  buildSvgChart,
  limitEventRows,
  limitTimelinePayload,
  updateTimelineLiveState,
} from "../chart.js";

export async function renderDemoPage(force = false) {
  const root = el.views.timeline;
  root.innerHTML = loadingCard("Loading case review…");

  try {
    const meta = state.cache.meta || (await api.meta());
    state.cache.meta = meta;

    const demos = await ensureDemoVideos(force);

    if (!state.timeline.selectedVideo && demos.length) {
      const firstReady = demos.find((v) => v.exists_in_inventory) || demos[0];
      state.timeline.selectedVideo = firstReady.video_id;
    }

    const selectedVideo = state.timeline.selectedVideo;
    const timeline = selectedVideo
      ? limitTimelinePayload(
          await ensureTimeline(selectedVideo, meta.selected_run_name, meta.final_experiment_name, force)
        )
      : null;
    const events = selectedVideo
      ? limitEventRows(
          await ensureVideoEvents(selectedVideo, meta.selected_run_name, force).catch(() => []),
          timeline?.fps || 0
        )
      : [];

    root.innerHTML = `
      <div class="grid">
        <section class="card">
          <div class="card__header">
            <h2 class="card__title">Select video</h2>
          </div>
          <div class="card__body split">
            <div class="field">
              <select id="timelineVideoSelect" class="select">
                ${demos
                  .map(
                    (v) =>
                      `<option value="${escapeHtml(v.video_id)}" ${
                        v.video_id === selectedVideo ? "selected" : ""
                      }>${escapeHtml(v.video_id)}</option>`
                  )
                  .join("")}
              </select>
              <div class="stat-strip">
                ${selectedVideo ? `<span class="pill pill--info mono">${escapeHtml(selectedVideo)}</span>` : ""}
                ${selectedVideo ? `<span class="pill ${events.length ? "pill--warn" : ""}">Known events: ${events.length}</span>` : ""}
                ${timeline ? `<span class="pill pill--info">Threshold: ${formatNumber(timeline.threshold, 3)}</span>` : ""}
              </div>
            </div>

            <div>
              <div class="video-panel" id="timelineVideoPanel">
                ${
                  selectedVideo
                    ? `<video id="timelineVideo" controls preload="metadata">
                         <source src="${escapeHtml(
                           api.demoVideoUrl(demos.find((v) => v.video_id === selectedVideo)?.url || "")
                         )}">
                       </video>`
                    : `<div class="empty-state">No demo video selected.</div>`
                }
                <div class="alert-overlay hidden" id="timelineAlertBadge">
                  <span class="pill pill--danger">Warning active</span>
                </div>
              </div>

              <div class="stat-strip" style="margin-top:12px;">
                <span class="pill" id="timelineCurrentTime">Current time: 0.00s</span>
                <span class="pill" id="timelineCurrentProb">Current score: 0.000</span>
                <span class="pill" id="timelineCurrentState">No warning</span>
                ${timeline ? `<span class="pill">Prediction horizon: ${escapeHtml(String(timeline.horizon_frames))} frames</span>` : ""}
              </div>
            </div>
          </div>
        </section>

        ${
          timeline
            ? `
        <section class="card chart-card">
          <div class="card__header">
            <div>
              <h2 class="card__title">Smoke warning score over time</h2>
              <ul class="card__subtitle">
                <li> Orange curve : model score </li>
                <li> Yellow dashed line : threshold </li>
                <li> Red dashed lines : known smoke starts </li>
                <li> White dashed line : current playback time </li>
              </ul>
            </div>
          </div>
          <div class="card__body">
            <div class="chart-wrap" id="timelineChart"></div>
          </div>
        </section>

        <section class="card">
          <div class="card__header"><h2 class="card__title">Timeline metadata</h2></div>
          <div class="card__body kv">
            <div class="kv__row"><div class="kv__key">Run</div><div class="kv__value mono">${escapeHtml(timeline.run_name)}</div></div>
            <div class="kv__row"><div class="kv__key">Experiment</div><div class="kv__value mono">${escapeHtml(timeline.exp_name)}</div></div>
            <div class="kv__row"><div class="kv__key">FPS</div><div class="kv__value">${escapeHtml(String(timeline.fps))}</div></div>
            <div class="kv__row"><div class="kv__key">Clip length</div><div class="kv__value">${escapeHtml(String(timeline.clip_len_frames))} frames</div></div>
            <div class="kv__row"><div class="kv__key">Prediction horizon</div><div class="kv__value">${escapeHtml(String(timeline.horizon_frames))} frames</div></div>
            <div class="kv__row"><div class="kv__key">Rows returned</div><div class="kv__value">${escapeHtml(String(timeline.rows.length))}</div></div>
          </div>
        </section>
        `
            : `<div class="empty-state card"><div>No timeline could be loaded for the selected video.</div></div>`
        }
      </div>
    `;

    const videoSelect = qs("#timelineVideoSelect", root);
    if (videoSelect) {
      videoSelect.addEventListener("change", async (e) => {
        state.timeline.selectedVideo = e.target.value;
        state.timeline.currentProb = 0;
        state.timeline.currentTime = 0;
        state.timeline.isAlert = false;
        await renderDemoPage(true);
      });
    }

    if (timeline) {
      const chartWrap = qs("#timelineChart", root);
      buildSvgChart({
        container: chartWrap,
        data: timeline.rows.map((r) => ({
          x: Number(r.t) / Number(timeline.fps),
          y: Number(r.prob),
        })),
        threshold: timeline.threshold,
        currentX: Math.min(state.timeline.currentTime, DEMO_MAX_SECONDS),
        eventXs: (timeline.event_starts_sec || []).map(Number),
      });
      updateTimelineLiveState(root, timeline);
    }
  } catch (error) {
    root.innerHTML = `<div class="card"><div class="card__body"><p class="muted">Could not load case review: ${escapeHtml(error.message)}</p></div></div>`;
  }
}
