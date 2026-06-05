/* Interactive results for "Do Machines Listen Like Humans?"
   Loads docs/data/*.json and renders four views with Plotly. */

// Candidate base directories for the data files, tried in order. The first is
// the document-relative path (works when served from docs/, or /docs/ with a
// trailing slash). The second resolves relative to this script's own URL, which
// also covers /docs without a trailing slash. Each is a fully-resolved absolute
// URL so fetch() never sees a malformed string.
const DATA_BASES = (() => {
  const bases = [];
  try { bases.push(new URL("data/", document.baseURI).href); } catch (e) {}
  try {
    const s = document.currentScript && document.currentScript.src;
    if (s) bases.push(new URL("data/", s).href);
  } catch (e) {}
  return bases.length ? bases : ["data/"];
})();

const COMPETITORS = ["Target", "Cohort", "Rhyme", "Unrelated", "Cross"];
const COMP_COLOR = {
  Target: "#4c8bf5", Cohort: "#e0524f", Rhyme: "#36b37e",
  Unrelated: "#8a93a3", Cross: "#b07cd6",
};
// When several competitor types are overlaid, color encodes the model and the
// dash pattern encodes the competitor type.
const COMP_DASH = { Target: "solid", Cohort: "dash", Rhyme: "dot", Unrelated: "dashdot", Cross: "longdash" };
const TYPE_COLOR = { causal: "#2ca58d", noncausal: "#e8943a", foundation: "#9b6dd6" };
const TYPE_LABEL = { causal: "Causal", noncausal: "Non-causal", foundation: "Foundation" };

// Distinct per-model palette (used when overlaying many models on one competitor)
const MODEL_PALETTE = [
  "#2ca58d", "#4c8bf5", "#36b37e", "#5ad1c0", "#7fb3ff", "#1f7a68",
  "#e8943a", "#e0524f", "#f2b441", "#c96f2e", "#d6485f",
  "#9b6dd6", "#b07cd6", "#7d5bb0",
];

const PLOT_FONT = { family: "inherit", color: "#cdd6e3", size: 12 };
const PLOT_BG = "rgba(0,0,0,0)";
const GRID = "#2a3242";

let DATA = null;     // trajectories.json
let METRICS = null;  // metrics.json
let MODEL_KEYS = [];
let MODEL_COLOR = {}; // key -> color

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG, font: PLOT_FONT,
    margin: { l: 56, r: 16, t: 16, b: 46 },
    xaxis: { title: "Time (ms)", gridcolor: GRID, zerolinecolor: GRID, color: "#9aa6b8" },
    yaxis: { title: "Activation", gridcolor: GRID, zerolinecolor: GRID, color: "#9aa6b8" },
    legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
    hovermode: "x unified",
    // namelength: -1 stops Plotly from truncating long model names (and the
    // value after them) in the hover box.
    hoverlabel: { namelength: -1, align: "left", bgcolor: "#1e2533",
                  bordercolor: "#2a3242", font: { color: "#e8edf4", size: 12 } },
  }, extra);
}
const CONFIG = { responsive: true, displayModeBar: false };

/* ------------------------------------------------------------------ */
/* Tabs                                                                */
/* ------------------------------------------------------------------ */
function initTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => activateView(btn.dataset.view));
  });
}
function activateView(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === name));
  // Plotly needs a resize nudge when a hidden div becomes visible
  window.dispatchEvent(new Event("resize"));
  if (name === "grid") renderGrid();
}

/* ------------------------------------------------------------------ */
/* Overview leaderboard                                                */
/* ------------------------------------------------------------------ */
function renderOverview() {
  const rows = MODEL_KEYS
    .map((k) => ({ k, name: METRICS.models[k].name, type: METRICS.models[k].type,
                   rmse: METRICS.models[k].metrics.Overall?.RMSE }))
    .filter((r) => r.rmse != null)
    .sort((a, b) => b.rmse - a.rmse); // worst at top so best ends at bottom-near-axis

  const trace = {
    type: "bar", orientation: "h",
    x: rows.map((r) => r.rmse),
    y: rows.map((r) => r.name),
    marker: { color: rows.map((r) => TYPE_COLOR[r.type]) },
    customdata: rows.map((r) => r.k),
    hovertemplate: "%{y}<br>Overall RMSE: %{x:.3f}<extra></extra>",
  };
  const layout = baseLayout({
    margin: { l: 150, r: 24, t: 10, b: 46 },
    xaxis: { title: "Overall RMSE vs. human", gridcolor: GRID, color: "#9aa6b8" },
    yaxis: { gridcolor: PLOT_BG, color: "#cdd6e3", automargin: true },
  });
  Plotly.newPlot("overview-chart", [trace], layout, CONFIG);
  $("#overview-chart").on("plotly_click", (ev) => {
    const key = ev.points[0].customdata;
    selectSingleModel(key);
    activateView("explorer");
  });
}

/* ------------------------------------------------------------------ */
/* Trajectory Explorer                                                 */
/* ------------------------------------------------------------------ */
const explorerState = { competitors: new Set(["Target"]), models: new Set(), human: true };

function initExplorer() {
  // competitor multi-select control
  const seg = $("#competitor-toggle");
  seg.classList.add("multi");
  COMPETITORS.forEach((c) => {
    const b = el("button", explorerState.competitors.has(c) ? "active" : "", c);
    b.addEventListener("click", () => {
      if (explorerState.competitors.has(c)) {
        // keep at least one competitor selected
        if (explorerState.competitors.size === 1) return;
        explorerState.competitors.delete(c);
      } else {
        explorerState.competitors.add(c);
      }
      b.classList.toggle("active", explorerState.competitors.has(c));
      renderExplorer();
    });
    seg.appendChild(b);
  });

  // model chips, grouped by type
  const box = $("#model-toggle");
  ["causal", "noncausal", "foundation"].forEach((type) => {
    MODEL_KEYS.filter((k) => DATA.models[k].type === type).forEach((k) => {
      const chip = el("span", "mchip");
      chip.dataset.key = k;
      const dot = el("i", "dot"); dot.style.background = MODEL_COLOR[k];
      chip.appendChild(dot);
      chip.appendChild(el("span", null, DATA.models[k].name));
      chip.addEventListener("click", () => toggleModel(k));
      box.appendChild(chip);
    });
  });

  $("#human-toggle").addEventListener("change", (e) => {
    explorerState.human = e.target.checked; renderExplorer();
  });
  $("#explorer-reset").addEventListener("click", () => {
    explorerState.models.clear();
    ["baseline", "noncausal-trans", "whisper"].forEach((k) => { if (DATA.models[k]) explorerState.models.add(k); });
    syncModelChips(); renderExplorer();
  });

  // default selection: one of each type tells the story
  ["baseline", "noncausal-trans", "whisper"].forEach((k) => { if (DATA.models[k]) explorerState.models.add(k); });
  syncModelChips();
  renderExplorer();
}

function toggleModel(k) {
  if (explorerState.models.has(k)) explorerState.models.delete(k);
  else explorerState.models.add(k);
  syncModelChips(); renderExplorer();
}
function selectSingleModel(k) {
  explorerState.models = new Set([k]);
  syncModelChips(); renderExplorer();
}
function syncModelChips() {
  document.querySelectorAll(".mchip").forEach((chip) => {
    const on = explorerState.models.has(chip.dataset.key);
    chip.classList.toggle("on", on);
    if (on) { chip.style.background = hexToRgba(MODEL_COLOR[chip.dataset.key], 0.16); chip.style.color = "#fff"; }
    else { chip.style.background = ""; chip.style.color = ""; }
  });
}

function renderExplorer() {
  const comps = COMPETITORS.filter((c) => explorerState.competitors.has(c)); // stable order
  const multi = comps.length > 1;
  const traces = [];
  [...explorerState.models].forEach((k) => {
    const m = DATA.models[k];
    comps.forEach((comp) => {
      if (!m.series[comp]) return;
      const label = multi ? `${m.name} · ${comp}` : m.name;
      traces.push({
        type: "scatter", mode: "lines", name: label,
        x: m.time_ms, y: m.series[comp],
        line: { color: MODEL_COLOR[k], width: 2.4, dash: COMP_DASH[comp] },
        hovertemplate: `${label}: %{y:.3f}<extra></extra>`,
      });
    });
  });
  if (explorerState.human) {
    comps.forEach((comp) => {
      if (!DATA.human.series[comp]) return;
      const label = multi ? `Human · ${comp}` : "Human (VWP)";
      traces.push({
        type: "scatter", mode: "lines", name: label,
        x: DATA.human.time_ms, y: DATA.human.series[comp],
        line: { color: "#ffffff", width: 2.6, dash: multi ? COMP_DASH[comp] : "dot" },
        hovertemplate: `${label}: %{y:.3f}<extra></extra>`,
      });
    });
  }
  const ytitle = multi ? "Activation" : `${comps[0]} activation`;
  const layout = baseLayout({ yaxis: { title: ytitle, gridcolor: GRID, color: "#9aa6b8" } });
  if (!traces.length) {
    layout.annotations = [{ text: "Select one or more models", showarrow: false,
      xref: "paper", yref: "paper", x: 0.5, y: 0.5, font: { color: "#93a1b5", size: 15 } }];
  }
  Plotly.react("explorer-chart", traces, layout, CONFIG);
}

/* ------------------------------------------------------------------ */
/* Model Grid (small multiples)                                        */
/* ------------------------------------------------------------------ */
const gridState = { competitors: new Set(["Target", "Cohort", "Rhyme"]), human: true, built: false };

function initGrid() {
  const seg = $("#grid-competitor-toggle");
  COMPETITORS.forEach((c) => {
    const b = el("button", gridState.competitors.has(c) ? "active" : "", c);
    b.addEventListener("click", () => {
      if (gridState.competitors.has(c)) gridState.competitors.delete(c);
      else gridState.competitors.add(c);
      b.classList.toggle("active");
      renderGrid(true);
    });
    seg.appendChild(b);
  });
  $("#grid-human-toggle").addEventListener("change", (e) => { gridState.human = e.target.checked; renderGrid(true); });
}

function renderGrid(force = false) {
  const wrap = $("#grid-wrap");
  if (!gridState.built || force) {
    wrap.innerHTML = "";
    gridState.built = true;
    MODEL_KEYS.forEach((k) => {
      const m = DATA.models[k];
      const card = el("div", "mini-card");
      const h = el("h3");
      h.appendChild(el("span", null, m.name));
      h.appendChild(el("span", `tag ${m.type}`, TYPE_LABEL[m.type]));
      card.appendChild(h);
      const plot = el("div", "mini-plot"); plot.id = `mini-${k}`;
      card.appendChild(plot);
      wrap.appendChild(card);

      const traces = [];
      [...gridState.competitors].forEach((c) => {
        if (!m.series[c]) return;
        traces.push({ type: "scatter", mode: "lines", name: c, x: m.time_ms, y: m.series[c],
          line: { color: COMP_COLOR[c], width: 2 }, hovertemplate: `${c}: %{y:.3f}<extra></extra>` });
        if (gridState.human && DATA.human.series[c]) {
          traces.push({ type: "scatter", mode: "lines", name: `${c} (human)`, showlegend: false,
            x: DATA.human.time_ms, y: DATA.human.series[c],
            line: { color: COMP_COLOR[c], width: 1.2, dash: "dot" }, opacity: 0.6,
            hovertemplate: `${c} human: %{y:.3f}<extra></extra>` });
        }
      });
      const layout = baseLayout({
        margin: { l: 40, r: 10, t: 6, b: 32 },
        xaxis: { gridcolor: GRID, color: "#9aa6b8", title: "" },
        yaxis: { gridcolor: GRID, color: "#9aa6b8", title: "" },
        showlegend: false, hovermode: "x unified",
      });
      Plotly.newPlot(plot.id, traces, layout, CONFIG);
    });
  }
}

/* ------------------------------------------------------------------ */
/* Metrics table                                                       */
/* ------------------------------------------------------------------ */
const metricState = { metric: "RMSE", sortCol: "Overall", asc: true };

function initMetrics() {
  const seg = $("#metric-toggle");
  ["RMSE", "MAE"].forEach((mt) => {
    const b = el("button", mt === metricState.metric ? "active" : "", mt);
    b.addEventListener("click", () => {
      metricState.metric = mt;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x.textContent === mt));
      metricState.asc = true; // lower is better for RMSE/MAE
      renderTable();
    });
    seg.appendChild(b);
  });
  renderTable();
}

function renderTable() {
  const cols = [...COMPETITORS, "Overall"];
  const table = $("#metrics-table");
  table.innerHTML = "";

  // header
  const thead = el("thead"); const trh = el("tr");
  trh.appendChild(el("th", null, "Model"));
  cols.forEach((c) => {
    const th = el("th", metricState.sortCol === c ? "sorted" + (metricState.asc ? " asc" : "") : "", c);
    th.addEventListener("click", () => {
      if (metricState.sortCol === c) metricState.asc = !metricState.asc;
      else { metricState.sortCol = c; metricState.asc = true; } // lower RMSE/MAE = better
      renderTable();
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh); table.appendChild(thead);

  // rows
  const rows = MODEL_KEYS.map((k) => {
    const mm = METRICS.models[k];
    const vals = {};
    cols.forEach((c) => { vals[c] = mm.metrics[c]?.[metricState.metric] ?? null; });
    return { k, name: mm.name, type: mm.type, vals };
  });
  const sc = metricState.sortCol;
  rows.sort((a, b) => {
    const av = a.vals[sc], bv = b.vals[sc];
    if (av == null) return 1; if (bv == null) return -1;
    return metricState.asc ? av - bv : bv - av;
  });

  // color scale bounds per visible metric for the Overall column heatmap
  const overallVals = rows.map((r) => r.vals.Overall).filter((v) => v != null);
  const lo = Math.min(...overallVals), hi = Math.max(...overallVals);

  const tbody = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr");
    tr.addEventListener("click", () => { selectSingleModel(r.k); activateView("explorer"); });
    const nameTd = el("td", "name");
    nameTd.appendChild(el("span", null, r.name));
    nameTd.appendChild(el("span", `pill ${r.type}`, TYPE_LABEL[r.type]));
    tr.appendChild(nameTd);
    cols.forEach((c) => {
      const v = r.vals[c];
      const td = el("td", null, v == null ? "—" : v.toFixed(3));
      if (c === "Overall" && v != null && hi > lo) {
        const t = (v - lo) / (hi - lo); // t=0 (lowest RMSE/MAE) -> green
        td.style.background = heat(t);
        td.style.color = "#07140d";
        td.style.fontWeight = "700";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// green (good) -> amber -> red (bad)
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[54, 179, 126], [240, 180, 65], [224, 82, 79]];
  const seg = t < 0.5 ? 0 : 1; const lt = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const a = stops[seg], b = stops[seg + 1];
  const mix = a.map((x, i) => Math.round(x + (b[i] - x) * lt));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

/* ------------------------------------------------------------------ */
/* Utils                                                               */
/* ------------------------------------------------------------------ */
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function assignColors() {
  // color by type but keep per-model distinguishable for multi-overlay
  let i = 0;
  MODEL_KEYS.forEach((k) => { MODEL_COLOR[k] = MODEL_PALETTE[i % MODEL_PALETTE.length]; i++; });
}

/* ------------------------------------------------------------------ */
/* Bootstrap                                                           */
/* ------------------------------------------------------------------ */
async function main() {
  // Preferred path: data baked into data.js as window.HTP_DATA (no fetch, so it
  // works over file:// and inside sandboxed webview previews).
  if (window.HTP_DATA && window.HTP_DATA.trajectories && window.HTP_DATA.metrics) {
    DATA = window.HTP_DATA.trajectories;
    METRICS = window.HTP_DATA.metrics;
  } else {
    // Fallback: fetch the JSON files (e.g. data.js missing but data/ present).
    const load = async (name) => {
      const errors = [];
      for (const base of DATA_BASES) {
        const url = base + name;
        try {
          const r = await fetch(url);
          if (!r.ok) { errors.push(`HTTP ${r.status} at ${url}`); continue; }
          return await r.json();
        } catch (e) {
          errors.push(`${e.message} at ${url}`);
        }
      }
      throw new Error(errors.join(" | "));
    };
    try {
      [DATA, METRICS] = await Promise.all([
        load("trajectories.json"),
        load("metrics.json"),
      ]);
    } catch (e) {
      document.querySelector("main").innerHTML =
        '<div class="card"><h2>Could not load data</h2>' +
        `<p class="muted">${e.message}</p>` +
        '<p class="muted">Rebuild the data bundle with <code>python docs/build_data.py</code> ' +
        '(it writes <code>docs/data.js</code>), then reload.</p></div>';
      return;
    }
  }
  // order: causal, noncausal, foundation
  const order = { causal: 0, noncausal: 1, foundation: 2 };
  MODEL_KEYS = Object.keys(DATA.models).sort((a, b) =>
    (order[DATA.models[a].type] - order[DATA.models[b].type]) || a.localeCompare(b));
  assignColors();

  initTabs();
  renderOverview();
  initExplorer();
  initGrid();
  initMetrics();

  // point the footer link at the GitHub repo this page is served from, if possible
  if (location.hostname.endsWith("github.io")) {
    const user = location.hostname.split(".")[0];
    const repo = location.pathname.split("/").filter(Boolean)[0] || "";
    if (repo) $("#repo-link").href = `https://github.com/${user}/${repo}`;
  }
}
main();
