import { state, DEMO_MAX_SECONDS } from "./state.js";
import { qs } from "./dom.js";
import { formatNumber } from "./format.js";

export function attachDemoTimeCap(videoEl, maxSeconds = DEMO_MAX_SECONDS) {
  if (!videoEl) return;

  const enforce = () => {
    if (Number(videoEl.currentTime || 0) >= maxSeconds) {
      videoEl.currentTime = maxSeconds;
      videoEl.pause();
    }
  };

  const handleSeek = () => {
    if (Number(videoEl.currentTime || 0) > maxSeconds) {
      videoEl.currentTime = maxSeconds;
    }
  };

  videoEl.addEventListener("timeupdate", enforce);
  videoEl.addEventListener("seeking", handleSeek);
  videoEl.addEventListener("loadedmetadata", handleSeek);
}

export function limitTimelinePayload(timeline, maxSeconds = DEMO_MAX_SECONDS) {
  if (!timeline) return null;
  return {
    ...timeline,
    rows: (timeline.rows || []).filter((r) => Number(r.t) / Number(timeline.fps) <= maxSeconds),
    event_starts_sec: (timeline.event_starts_sec || []).map(Number).filter((x) => x <= maxSeconds),
  };
}

export function limitEventRows(events, fps, maxSeconds = DEMO_MAX_SECONDS) {
  if (!Array.isArray(events)) return [];
  if (!fps) return events;
  const maxFrames = Number(fps) * maxSeconds;
  return events.filter((row) => Number(row.event_start) <= maxFrames);
}

export function nearestTimelineRow(rows, currentSec, fps) {
  if (!rows?.length) return null;
  let best = rows[0];
  let bestDelta = Infinity;
  for (const row of rows) {
    const sec = Number(row.t) / Number(fps);
    const delta = Math.abs(sec - currentSec);
    if (delta < bestDelta) {
      best = row;
      bestDelta = delta;
    }
  }
  return best;
}

export function updateTimelineLiveState(root, timeline) {
  const video = qs("#timelineVideo", root);
  const badge = qs("#timelineAlertBadge", root);
  const panel = qs("#timelineVideoPanel", root);
  const currentTimePill = qs("#timelineCurrentTime", root);
  const currentProbPill = qs("#timelineCurrentProb", root);
  const currentStatePill = qs("#timelineCurrentState", root);
  const chartWrap = qs("#timelineChart", root);

  if (!video || !timeline) return;

  attachDemoTimeCap(video);

  const sync = () => {
    const currentSec = Math.min(Number(video.currentTime || 0), DEMO_MAX_SECONDS);
    state.timeline.currentTime = currentSec;
    const row = nearestTimelineRow(timeline.rows, currentSec, timeline.fps);
    const prob = row ? Number(row.prob) : 0;
    const isAlert = row ? Number(row.pred) === 1 : false;

    state.timeline.currentProb = prob;
    state.timeline.isAlert = isAlert;

    if (currentTimePill) currentTimePill.textContent = `Current time: ${formatNumber(currentSec, 2)}s`;
    if (currentProbPill) currentProbPill.textContent = `Current score: ${formatNumber(prob, 3)}`;
    if (currentStatePill) currentStatePill.textContent = isAlert ? "Warning active" : "No warning";

    if (badge) badge.classList.toggle("hidden", !isAlert);
    if (panel) panel.classList.toggle("alert-active", isAlert);

    buildSvgChart({
      container: chartWrap,
      data: timeline.rows.map((r) => ({
        x: Number(r.t) / Number(timeline.fps),
        y: Number(r.prob),
      })),
      threshold: timeline.threshold,
      currentX: currentSec,
      eventXs: (timeline.event_starts_sec || []).map(Number),
    });
  };

  video.addEventListener("timeupdate", sync);
  video.addEventListener("seeked", sync);
  video.addEventListener("loadedmetadata", sync);
  sync();
}

export function buildSvgChart({
  container,
  data,
  threshold = null,
  currentX = null,
  eventXs = [],
  width = 1000,
  height = 320,
}) {
  if (!container) return;
  container.innerHTML = "";

  if (!data?.length) {
    container.innerHTML = `<div class="empty-state"><div>No chart data available.</div></div>`;
    return;
  }

  const pad = { top: 18, right: 16, bottom: 36, left: 46 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const minX = Math.min(...data.map((d) => d.x));
  const maxX = Math.max(...data.map((d) => d.x));
  const rangeX = maxX - minX || 1;

  const xScale = (x) => pad.left + ((x - minX) / rangeX) * innerW;
  const yScale = (y) => pad.top + (1 - y) * innerH;

  const linePath = data
    .map((d, i) => `${i === 0 ? "M" : "L"}${xScale(d.x).toFixed(2)},${yScale(d.y).toFixed(2)}`)
    .join(" ");

  const thresholdY = threshold == null ? null : yScale(Number(threshold)).toFixed(2);

  const eventLines = eventXs
    .map((x) => {
      const sx = xScale(x).toFixed(2);
      return `<line x1="${sx}" y1="${pad.top}" x2="${sx}" y2="${height - pad.bottom}" stroke="#ff6b57" stroke-width="1.5" stroke-dasharray="5 5" />`;
    })
    .join("");

  const currentLine =
    currentX == null
      ? ""
      : (() => {
          const sx = xScale(currentX).toFixed(2);
          return `<line x1="${sx}" y1="${pad.top}" x2="${sx}" y2="${height - pad.bottom}" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="6 6" />`;
        })();

  const xTicks = Array.from({ length: 6 }, (_, i) => minX + (rangeX * i) / 5);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  container.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
      ${yTicks
        .map((y) => {
          const sy = yScale(y).toFixed(2);
          return `
            <line x1="${pad.left}" y1="${sy}" x2="${width - pad.right}" y2="${sy}" stroke="rgba(255,255,255,0.08)" />
            <text x="${pad.left - 8}" y="${Number(sy) + 4}" font-size="11" fill="rgba(255,255,255,0.55)" text-anchor="end">${y.toFixed(2)}</text>
          `;
        })
        .join("")}
      ${xTicks
        .map((x) => {
          const sx = xScale(x).toFixed(2);
          return `
            <line x1="${sx}" y1="${pad.top}" x2="${sx}" y2="${height - pad.bottom}" stroke="rgba(255,255,255,0.05)" />
            <text x="${sx}" y="${height - 12}" font-size="11" fill="rgba(255,255,255,0.55)" text-anchor="middle">${x.toFixed(1)}s</text>
          `;
        })
        .join("")}
      ${eventLines}
      ${thresholdY ? `<line x1="${pad.left}" y1="${thresholdY}" x2="${width - pad.right}" y2="${thresholdY}" stroke="#ffb347" stroke-width="1.5" stroke-dasharray="6 6" />` : ""}
      ${currentLine}
      <path d="${linePath}" fill="none" stroke="#ff8a1f" stroke-width="2.5" />
    </svg>
  `;
}
