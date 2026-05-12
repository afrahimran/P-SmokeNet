import { el } from "./dom.js";

export function setBanner(message = "", isError = false) {
  if (!message) {
    el.globalBanner.classList.add("hidden");
    el.globalBanner.textContent = "";
    return;
  }
  el.globalBanner.classList.remove("hidden");
  el.globalBanner.textContent = message;
  el.globalBanner.style.background = isError
    ? "rgba(255, 107, 87, 0.08)"
    : "rgba(255, 138, 31, 0.08)";
  el.globalBanner.style.borderColor = isError
    ? "rgba(255, 107, 87, 0.24)"
    : "rgba(255, 138, 31, 0.24)";
  el.globalBanner.style.color = isError ? "#ffd6cf" : "#ffd7b0";
}

export function setBackendStatus(ok, label) {
  el.backendStatusPill.textContent = label;
  el.backendStatusPill.className = `pill ${ok ? "pill--success" : "pill--danger"}`;
}
