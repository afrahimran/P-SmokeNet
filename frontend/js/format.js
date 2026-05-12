export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function formatNumber(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

export function formatPercent(value, digits = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "—";
}

export function loadingCard(text = "Loading…") {
  const tpl = document.getElementById("loadingCardTemplate");
  if (!tpl) {
    return `<div class="card card--soft centered-card"><p class="muted">${escapeHtml(text)}</p></div>`;
  }
  const node = tpl.content.cloneNode(true);
  const p = node.querySelector("p");
  if (p) p.textContent = text;
  const wrap = document.createElement("div");
  wrap.appendChild(node);
  return wrap.innerHTML;
}
