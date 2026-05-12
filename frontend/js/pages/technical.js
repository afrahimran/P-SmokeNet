import { el, qs } from "../dom.js";
import { state } from "../state.js";
import { escapeHtml, formatNumber, formatPercent, loadingCard } from "../format.js";
import {
  api,
  ensureComparisons,
  ensureExperimentSummary,
  ensureExperiments,
  ensureRunDetail,
  ensureRuns,
} from "../api.js";

function renderComparisonTable(rows) {
  const sorted = [...rows].sort((a, b) => {
    const af = String(a.exp_name || "").toLowerCase().includes("fusion");
    const bf = String(b.exp_name || "").toLowerCase().includes("fusion");
    if (af && !bf) return -1;
    if (!af && bf) return 1;
    return String(a.exp_name || "").localeCompare(String(b.exp_name || ""));
  });

  return `
    <div class="table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Backbone</th>
            <th>Type</th>
            <th>Physics</th>
            <th>F1</th>
            <th>PR-AUC</th>
            <th>Detection</th>
            <th>Lead Time</th>
            <th>False Alarms</th>
          </tr>
        </thead>
        <tbody>
          ${sorted
            .map(
              (r) => `
              <tr>
                <td>${escapeHtml(r.exp_name || "—")}</td>
                <td>${escapeHtml(r.backbone_name || "—")}</td>
                <td>${escapeHtml(r.model_type || "—")}</td>
                <td>${escapeHtml(String(r.use_physics))}</td>
                <td>${formatNumber(r.test_f1, 3)}</td>
                <td>${formatNumber(r.test_pr_auc, 3)}</td>
                <td>${formatPercent(r.event_detection_rate, 0)}</td>
                <td>${formatNumber(r.mean_lead_time_sec, 2)}s</td>
                <td>${formatNumber(r.false_alarms_per_min, 2)}</td>
              </tr>
            `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

export async function renderTechnicalPage(force = false) {
  const root = el.views.runs;
  root.innerHTML = loadingCard("Loading technical details…");

  try {
    const meta = state.cache.meta || (await api.meta());
    state.cache.meta = meta;

    const runName = state.technical.selectedRun || meta.selected_run_name;
    const [runs, runDetail, experiments, comparisons] = await Promise.all([
      ensureRuns(force),
      ensureRunDetail(runName, force),
      ensureExperiments(runName, force),
      ensureComparisons(force),
    ]);

    if (!state.technical.selectedExperiment) {
      const fusion =
        experiments.find((x) => String(x.exp_name || "").toLowerCase().includes("fusion")) ||
        experiments.find((x) => x.exp_name === meta.final_experiment_name);
      state.technical.selectedExperiment = fusion?.exp_name || experiments[0]?.exp_name || "";
    }

    const expName = state.technical.selectedExperiment;
    const summary = expName
      ? await ensureExperimentSummary(runName, expName, force).catch(() => null)
      : null;

    root.innerHTML = `
      <div class="grid">
        <section class="card">
          <div class="card__header">
            <div>
              <h2 class="card__title">Run and experiment selection</h2>
              <p class="card__subtitle">Fusion experiment opens first when available.</p>
            </div>
          </div>
          <div class="card__body">
            <div class="toolbar">
              <div class="field" style="min-width:220px;">
                <label for="runSelect">Run</label>
                <select id="runSelect" class="select">
                  ${runs
                    .map(
                      (r) =>
                        `<option value="${escapeHtml(r)}" ${r === runName ? "selected" : ""}>${escapeHtml(r)}</option>`
                    )
                    .join("")}
                </select>
              </div>

              <div class="field" style="min-width:360px; flex:1;">
                <label for="experimentSelect">Experiment</label>
                <select id="experimentSelect" class="select">
                  ${experiments
                    .map(
                      (x) =>
                        `<option value="${escapeHtml(x.exp_name)}" ${
                          x.exp_name === expName ? "selected" : ""
                        }>${escapeHtml(x.exp_name)}</option>`
                    )
                    .join("")}
                </select>
              </div>
            </div>
          </div>
        </section>

        ${
          summary
            ? `
        <section class="card">
          <div class="card__header">
            <div>
              <h2 class="card__title">Experiment summary</h2>
              <p class="card__subtitle">Summary.json values for the selected experiment.</p>
            </div>
          </div>
          <div class="card__body">
            <pre class="code-block">${escapeHtml(JSON.stringify(summary, null, 2))}</pre>
          </div>
        </section>
        `
            : ""
        }

        <section class="card">
          <div class="card__header">
            <div>
              <h2 class="card__title">Comparison studies</h2>
              <p class="card__subtitle">Saved CSV comparison artifacts from the comparisons folder.</p>
            </div>
          </div>
          <div class="card__body">
            ${
              comparisons.length
                ? comparisons
                    .map(
                      (group, index) => `
                  <details class="comparison-block" ${index === 0 ? "open" : ""}>
                    <summary class="comparison-summary">${escapeHtml(group.label || group.fileName)}</summary>
                    <div style="margin-top: 12px;">
                      <div class="muted small" style="margin-bottom: 10px;">${escapeHtml(group.path || "")}</div>
                      ${renderComparisonTable(group.rows || [])}
                    </div>
                  </details>
                `
                    )
                    .join("")
                : `<div class="empty-state"><div>No comparison CSVs were found.</div></div>`
            }
          </div>
        </section>
      </div>
    `;

    qs("#runSelect", root)?.addEventListener("change", async (e) => {
      state.technical.selectedRun = e.target.value;
      state.technical.selectedExperiment = "";
      await renderTechnicalPage(true);
    });

    qs("#experimentSelect", root)?.addEventListener("change", async (e) => {
      state.technical.selectedExperiment = e.target.value;
      await renderTechnicalPage(true);
    });
  } catch (error) {
    root.innerHTML = `<div class="card"><div class="card__body"><p class="muted">Could not load technical view: ${escapeHtml(error.message)}</p></div></div>`;
  }
}
