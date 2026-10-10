/* BayesLab: a dependency-free interface to the exact Python inference engine. */
"use strict";

const icons = {
  network:
    '<circle cx="6" cy="6" r="3"/><circle cx="18" cy="8" r="3"/><circle cx="10" cy="19" r="3"/>' +
    '<path d="m9 6 6 1M7 9l2 7m7-5-4 6"/>',
  grid:
    '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>' +
    '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  umbrella: '<path d="M3 12a9 9 0 0 1 18 0H3Zm9-9v16a2 2 0 0 0 4 0M12 3c-4 3-4 6-4 9m4-9c4 3 4 6 4 9"/>',
  alarm: '<path d="M8 20h8m-4-2V5M5 18h14l-2-4V9a5 5 0 0 0-10 0v5l-2 4ZM10 3h4"/>',
  leaf: '<path d="M20 3C7 3 2 9 5 16s15 6 15-13Z M5 20 16 9m-6 6v-5m0 5h5"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/>',
  chart: '<path d="M4 3v17h17M8 15v-4m5 4V6m5 9V9"/>',
  diamond: '<path d="m12 2 10 10-10 10L2 12 10-10Z"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  spark: '<path d="m12 2 2.5 7.5L22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5L12 2Z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 10v7m0-11v1"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"` +
  ` stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.network}</svg>`;
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
// Both formatters return safe HTML for any input: drafts and imported files may hold text where
// numbers belong, and that text is escaped rather than trusted.
const pct = (value) =>
  typeof value !== "number" || !Number.isFinite(value)
    ? escapeHtml(value ?? "—")
    : value > 0 && value < 0.0001
      ? value * 100 >= 0.000001
        ? (100 * value).toLocaleString(undefined, { maximumSignificantDigits: 4 })
        : (100 * value).toExponential(3)
      : (100 * value).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });
// A probability too small for a float (below about 1e-308) arrives as 0 plus its log10.
const probabilityText = (value, log10) => {
  if (value >= 1e-300 || !Number.isFinite(log10)) return value.toPrecision(8);
  let exponent = Math.floor(log10);
  let mantissa = 10 ** (log10 - exponent);
  if (mantissa >= 9.99999995) [mantissa, exponent] = [1, exponent + 1];
  return `${mantissa.toFixed(7)}e${exponent}`;
};
const num = (value) =>
  typeof value !== "number" || !Number.isFinite(value)
    ? escapeHtml(value ?? "—")
    : Math.abs(value) > 0 && Math.abs(value) < 0.0001
      ? value.toExponential(4)
      : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const assignmentText = (values) =>
  Object.entries(values)
    .map(([name, value]) => `${name} = ${value}`)
    .join(", ");
const state = {
  examples: [],
  model: null,
  example: "umbrella",
  selected: "Rain",
  tab: "probability",
  mode: "conditional",
  event: {},
  evidence: {},
  decisionEvidence: {},
  action: null,
  query: null,
  utility: null,
  error: "",
  busy: false,
  draft: "",
  editorMessage: "",
  request: 0,
  method: "elimination",
  worlds: null,
  sensitivity: null,
  utilityDraft: [],
  costDraft: {},
  config: { public: false },
  editorView: "builder",
  builder: null,
  builderDirty: false,
  graphZoom: null,
};
const app = document.getElementById("app");
let toastTimer;
const SESSION_KEY = "bayeslab.session.v1";

async function api(path, body) {
  let response;
  try {
    response = await fetch(
      path,
      body === undefined
        ? {}
        : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    );
  } catch {
    throw Object.assign(
      new Error("Cannot reach the BayesLab server. Check that it is still running, then try again."),
      { status: 0 },
    );
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    // Not JSON: a proxy error page, for example.
  }
  if (!response.ok || data === null)
    throw Object.assign(
      new Error(data?.error || `The server sent an unexpected reply (HTTP ${response.status}). Try again.`),
      { status: response.status },
    );
  return data;
}

function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("visible"), 3500);
}

function setModel(model, example = "custom") {
  model = {
    ...model,
    name: model.name ?? "Untitled network",
    chance_nodes: model.chance_nodes.map((node) => ({ ...node, parents: node.parents || [] })),
    decision_nodes: (model.decision_nodes || []).map((node) => ({
      ...node,
      information: node.information || [],
    })),
    utility_nodes: (model.utility_nodes || []).map((node) => ({ ...node, parents: node.parents || [] })),
  };
  state.request++;
  const names = [...model.chance_nodes, ...model.decision_nodes, ...model.utility_nodes].map((node) => node.name);
  Object.assign(state, {
    model,
    example,
    selected: names.includes(state.selected) ? state.selected : model.chance_nodes[0].name,
    event: { [model.chance_nodes[0].name]: model.chance_nodes[0].states[0] },
    evidence: {},
    decisionEvidence: {},
    action: model.decision_nodes?.[0]?.states[0] ?? null,
    query: null,
    utility: null,
    busy: false,
    error: "",
    draft: JSON.stringify(model, null, 2),
    editorMessage: "",
    mode: "marginal",
    worlds: null,
    sensitivity: null,
    utilityDraft: structuredClone(model.utility_nodes),
    costDraft: { ...(model.decision_nodes[0]?.observation_costs || {}) },
    builder: null,
    builderDirty: false,
    graphZoom: null,
  });
  if (example === "umbrella") {
    state.mode = "conditional";
    state.evidence = { Forecast: "rainy" };
    state.decisionEvidence = { Forecast: "rainy" };
  } else if (example === "alarm") {
    state.mode = "conditional";
    state.evidence = { JohnCalls: "yes", MaryCalls: "yes" };
  } else if (example === "sprinkler") {
    state.mode = "conditional";
    state.event = { Rain: "yes" };
    state.evidence = { WetGrass: "yes" };
  }
  if (state.tab === "utility" && !model.decision_nodes?.length) state.tab = "probability";
  render();
}

async function loadExample(id) {
  const token = ++state.request;
  try {
    const model = await api(`/api/examples/${encodeURIComponent(id)}`);
    if (token !== state.request) return;
    setModel(model, id);
    if (state.tab !== "model") await calculate();
  } catch (error) {
    toast(error.message);
  }
}

// While the visual builder is open, the diagram previews its unapplied draft.
function graphModel() {
  return state.tab === "model" && state.editorView === "builder" && state.builder ? state.builder : state.model;
}

function renderGraphCard() {
  const card = document.getElementById("graph-card");
  if (!card) return;
  const model = graphModel();
  const decisions = model.decision_nodes?.length || 0;
  const utilities = model.utility_nodes?.length || 0;
  const draft =
    model === state.builder && state.builderDirty
      ? '<span class="draft-badge" title="Validate & apply to use this draft">Unapplied draft</span>'
      : "";
  card.innerHTML = `
    <div class="card-header">
      <div class="card-title">
        ${icon("network")}<h2>${escapeHtml(model.name)}</h2>
        <span class="count-badge">${model.chance_nodes.length + decisions + utilities} nodes</span>${draft}
      </div>
      <div class="legend">
        <span><i></i>Chance</span>
        <span><i class="decision"></i>Decision</span>
        <span><i class="utility"></i>Utility</span>
      </div>
    </div>
    <div class="graph-layout">
      <div class="graph-area" id="graph"></div>
      <aside class="node-inspector" id="inspector" aria-label="Selected node details"></aside>
    </div>
    <div class="graph-footer">
      <span><strong>${model.chance_nodes.length}</strong> chance variables</span>
      <span><strong>${decisions}</strong> decision</span>
      <span><strong>${utilities}</strong> utility</span>
      <span>Directed acyclic graph</span>
    </div>`;
  renderGraph();
  renderInspector();
}

function render() {
  const model = state.model;
  if (!model) return;
  const focusKey = document.activeElement?.dataset?.focus;
  const exampleButtons = state.examples
    .map((example) => {
      const active = state.example === example.id;
      const glyph = { umbrella: "umbrella", alarm: "alarm", sprinkler: "leaf" }[example.id];
      return `
        <button class="network-button ${active ? "active" : ""}" data-example="${escapeHtml(example.id)}"
          ${active ? 'aria-current="true"' : ""}>
          <span class="example-icon">${icon(glyph)}</span>
          <span><strong>${escapeHtml(example.name)}</strong><small>${escapeHtml(example.category)}</small></span>
        </button>`;
    })
    .join("");
  const tabs = [
    ["probability", "chart", "Probability inference"],
    ["utility", "diamond", "Decision & utility"],
    ["model", "code", "Model data"],
  ]
    .map(([id, glyph, title]) => {
      const active = state.tab === id;
      return `
          <button class="tab ${active ? "active" : ""}" data-tab="${id}"
            aria-pressed="${active}">${icon(glyph)} ${title}</button>`;
    })
    .join("");
  const footerNote =
    state.tab === "utility"
      ? "Utilities are model-defined preference scores."
      : "Discrete states · exact calculations · transparent steps";
  app.innerHTML = `<div class="shell">
    <aside class="sidebar" aria-label="Workspace navigation">
      <div class="brand">
        <div class="brand-mark">${icon("network")}</div>
        <div>Bayes<span>Lab</span></div>
      </div>
      <div>
        <div class="sidebar-label">Workspace</div>
        <div class="nav-item">${icon("grid")} Network playground</div>
      </div>
      <div>
        <div class="sidebar-label">Explore a network</div>
        <nav class="network-list" aria-label="Example networks">${exampleButtons}</nav>
      </div>
      <div class="sidebar-note">
        ${
          state.config.public
            ? `<strong><span class="status-dot"></span> An online thinking space</strong>
        <p>Calculations run on this BayesLab server, which does not store your models. Your work is saved
          only in this browser. No account needed.</p>`
            : `<strong><span class="status-dot"></span> A local thinking space</strong>
        <p>Every calculation runs on your computer, and your work is saved in this browser. No account or API key
          needed.</p>`
        }
      </div>
      <div class="sidebar-footer">BAYESLAB <span> / </span> v${escapeHtml(state.config.version || "1.2")}</div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div class="breadcrumb"><span>Workspace</span><span>/</span><strong>Network playground</strong></div>
        <div class="top-actions">
          <button class="button secondary" data-command="edit">${icon("code")} Edit model</button>
          <button class="button secondary" data-command="export">${icon("download")} Export JSON</button>
        </div>
      </header>
      <div class="content">
        <section class="intro">
          <div>
            <div class="eyebrow">Probability, with perspective</div>
            <h1>Reason under uncertainty.</h1>
            <p>Explore what’s likely. Understand what matters. Make a better decision.</p>
          </div>
          <div class="exact-pill">${icon("spark")} Exact inference engine</div>
        </section>
        <section class="card graph-card" id="graph-card" aria-label="Network diagram"></section>
        <section class="card analysis-card" aria-label="Analysis workbench">
          <nav class="tabs" aria-label="Analysis views">${tabs}</nav>
          <div id="workbench"></div>
          <div class="panel-footer">
            <span>${icon("info")} ${footerNote}</span><span>Enumeration & variable elimination</span>
          </div>
        </section>
        <p class="bottom-note">
          Bayesian Network Probability and Utility Determination · Built for curiosity, learning, and better decisions.
        </p>
      </div>
    </main>
  </div>`;
  renderGraphCard();
  renderWorkbench();
  restoreFocus(focusKey);
}

// Put keyboard focus back on the control that had it before a re-render.
function restoreFocus(key) {
  if (!key) return;
  const target = [...app.querySelectorAll("[data-focus]")].find((item) => item.dataset.focus === key);
  target?.focus();
}

function graphNodes(model = graphModel()) {
  return [
    ...model.chance_nodes.map((node) => ({ ...node, type: "chance" })),
    ...(model.decision_nodes || []).map((node) => ({
      ...node,
      parents: node.information || [],
      type: "decision",
    })),
    ...(model.utility_nodes || []).map((node) => ({ ...node, type: "utility" })),
  ];
}

const GRAPH = { layerGap: 170, nodeGap: 92, dummyGap: 22, marginX: 74, marginY: 40, halfWidth: 59 };
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 2.5;

/* Layered (Sugiyama-style) layout: longest-path layers, a waypoint in every layer
   a long arrow crosses, barycenter sweeps to reduce crossings, then vertical
   positions pulled toward neighbours while keeping nodes apart. */
function layoutGraph(nodes) {
  const known = new Set(nodes.map((node) => node.name));
  const parentsOf = (node) => (node.parents || []).filter((name) => known.has(name));
  const depth = new Map();
  let progress = true;
  while (depth.size < nodes.length && progress) {
    progress = false;
    for (const node of nodes)
      if (!depth.has(node.name) && parentsOf(node).every((name) => depth.has(name))) {
        const parents = parentsOf(node);
        depth.set(node.name, parents.length ? Math.max(...parents.map((name) => depth.get(name))) + 1 : 0);
        progress = true;
      }
  }
  for (const node of nodes) if (!depth.has(node.name)) depth.set(node.name, 0);
  const layerCount = Math.max(0, ...depth.values()) + 1;
  const order = Array.from({ length: layerCount }, () => []);
  const layerOf = new Map();
  const dummy = new Set();
  for (const node of nodes) {
    order[depth.get(node.name)].push(node.name);
    layerOf.set(node.name, depth.get(node.name));
  }
  const links = [];
  const edges = [];
  let next = 0;
  for (const node of nodes)
    for (const parent of parentsOf(node)) {
      const chain = [parent];
      for (let layer = depth.get(parent) + 1; layer < depth.get(node.name); layer++) {
        const id = `\u0000${next++}`;
        dummy.add(id);
        order[layer].push(id);
        layerOf.set(id, layer);
        chain.push(id);
      }
      chain.push(node.name);
      for (let i = 0; i + 1 < chain.length; i++) links.push([chain[i], chain[i + 1]]);
      edges.push({ chain, observation: node.type === "decision" });
    }
  const above = new Map();
  const below = new Map();
  for (const [from, to] of links) {
    below.set(from, [...(below.get(from) || []), to]);
    above.set(to, [...(above.get(to) || []), from]);
  }
  const positions = () => {
    const map = new Map();
    order.forEach((ids) => ids.forEach((id, i) => map.set(id, i)));
    return map;
  };
  const crossings = () => {
    const pos = positions();
    let total = 0;
    for (let layer = 0; layer + 1 < layerCount; layer++) {
      const segments = links.filter(([from]) => layerOf.get(from) === layer).map(([a, b]) => [pos.get(a), pos.get(b)]);
      for (let i = 0; i < segments.length; i++)
        for (let j = i + 1; j < segments.length; j++)
          if ((segments[i][0] - segments[j][0]) * (segments[i][1] - segments[j][1]) < 0) total++;
    }
    return total;
  };
  let best = order.map((ids) => [...ids]);
  let fewest = crossings();
  for (let sweep = 0; sweep < 16 && fewest > 0; sweep++) {
    const downward = sweep % 2 === 0;
    const layers = [...Array(layerCount).keys()];
    for (const layer of downward ? layers.slice(1) : layers.slice(0, -1).reverse()) {
      const pos = positions();
      const neighbours = downward ? above : below;
      const keyed = order[layer].map((id, i) => {
        const list = neighbours.get(id) || [];
        return { id, i, key: list.length ? list.reduce((sum, item) => sum + pos.get(item), 0) / list.length : i };
      });
      keyed.sort((a, b) => a.key - b.key || a.i - b.i);
      order[layer] = keyed.map((item) => item.id);
    }
    const count = crossings();
    if (count < fewest) {
      fewest = count;
      best = order.map((ids) => [...ids]);
    }
  }
  best.forEach((ids, layer) => (order[layer] = ids));
  const size = (id) => (dummy.has(id) ? GRAPH.dummyGap : GRAPH.nodeGap);
  const y = new Map();
  for (const ids of order) {
    let offset = -ids.reduce((sum, id) => sum + size(id), 0) / 2;
    for (const id of ids) {
      y.set(id, offset + size(id) / 2);
      offset += size(id);
    }
  }
  // Pull each node toward the average height of its neighbours, keeping order and spacing.
  const place = (ids, neighbours) => {
    const want = ids.map((id) => {
      const list = neighbours(id);
      return list.length ? list.reduce((sum, item) => sum + y.get(item), 0) / list.length : y.get(id);
    });
    const gap = (i) => (size(ids[i - 1]) + size(ids[i])) / 2;
    const forward = [...want];
    for (let i = 1; i < ids.length; i++) forward[i] = Math.max(want[i], forward[i - 1] + gap(i));
    const backward = [...want];
    for (let i = ids.length - 2; i >= 0; i--) backward[i] = Math.min(want[i], backward[i + 1] - gap(i + 1));
    ids.forEach((id, i) => y.set(id, (forward[i] + backward[i]) / 2));
  };
  for (let pass = 0; pass < 8; pass++) {
    const layers = [...Array(layerCount).keys()];
    if (pass % 2 === 0) for (const layer of layers.slice(1)) place(order[layer], (id) => above.get(id) || []);
    else for (const layer of layers.slice(0, -1).reverse()) place(order[layer], (id) => below.get(id) || []);
  }
  for (const layer of [...Array(layerCount).keys()])
    place(order[layer], (id) => [...(above.get(id) || []), ...(below.get(id) || [])]);
  const top = Math.min(...[...y].map(([id, value]) => value - size(id) / 2));
  const bottom = Math.max(...[...y].map(([id, value]) => value + size(id) / 2));
  const point = (id) => ({ x: GRAPH.marginX + layerOf.get(id) * GRAPH.layerGap, y: y.get(id) - top + GRAPH.marginY });
  const position = new Map(nodes.map((node) => [node.name, point(node.name)]));
  const curve = (from, to) => {
    const middle = (from.x + to.x) / 2;
    return ` C ${middle} ${from.y}, ${middle} ${to.y}, ${to.x} ${to.y}`;
  };
  const paths = edges.map(({ chain, observation }) => {
    const start = point(chain[0]);
    let current = { x: start.x + GRAPH.halfWidth, y: start.y };
    let d = `M ${current.x} ${current.y}`;
    for (const id of chain.slice(1, -1)) {
      const waypoint = point(id);
      // Cross the layer horizontally, through the gap between its nodes.
      const enter = { x: waypoint.x - GRAPH.halfWidth, y: waypoint.y };
      d += curve(current, enter) + ` L ${waypoint.x + GRAPH.halfWidth} ${waypoint.y}`;
      current = { x: waypoint.x + GRAPH.halfWidth, y: waypoint.y };
    }
    const end = point(chain[chain.length - 1]);
    d += curve(current, { x: end.x - GRAPH.halfWidth - 4, y: end.y });
    return { d, observation };
  });
  return {
    position,
    paths,
    width: Math.max(2 * GRAPH.marginX + (layerCount - 1) * GRAPH.layerGap, 260),
    height: bottom - top + 2 * GRAPH.marginY,
  };
}

function renderGraph() {
  const element = document.getElementById("graph");
  if (!element) return;
  const model = graphModel();
  const nodes = graphNodes(model);
  if (!nodes.length) {
    element.innerHTML = '<p class="graph-empty">No nodes yet. Add a chance variable in the builder below.</p>';
    return;
  }
  if (!nodes.some((node) => node.name === state.selected)) state.selected = nodes[0].name;
  const { position, paths, width, height } = layoutGraph(nodes);
  const edges = paths
    .map(
      ({ d, observation }) =>
        `<path d="${d}" fill="none" stroke="${observation ? "#b8a5d9" : "#cac4d8"}" stroke-width="1.4"` +
        `${observation ? ' stroke-dasharray="4 4"' : ""} marker-end="url(#arrowhead)"/>`,
    )
    .join("");
  const nodeMarkup = nodes
    .map((node, index) => {
      const { x, y } = position.get(node.name);
      const selected = state.selected === node.name;
      const color = { chance: "#6b8cd1", decision: "#9b7ad6", utility: "#cea665" }[node.type];
      // Darker shades of the same hues keep the type symbols readable on the light node fills.
      const ink = { chance: "#426cc4", decision: "#7f55cb", utility: "#906b2e" }[node.type];
      const fill = { chance: "#f2f6ff", decision: "#f7f2ff", utility: "#fffaf0" }[node.type];
      const symbol = node.type === "chance" ? "○" : node.type === "decision" ? "□" : "◇";
      const label = node.name.length > 16 ? node.name.slice(0, 14) + "…" : node.name;
      const unit = node.type === "decision" ? "actions" : "states";
      const caption = node.type === "utility" ? "utility" : `${node.states.length} ${unit}`;
      const halo = selected
        ? `<rect x="-64" y="-34" width="128" height="68" rx="${node.type === "decision" ? 11 : 17}" fill="none"` +
          ` stroke="${color}" opacity=".13" stroke-width="5"/>`
        : "";
      return `
        <g class="node-group" data-node-index="${index}" tabindex="0" role="button" data-focus="graph:${escapeHtml(node.name)}"
          aria-label="Inspect ${escapeHtml(node.name)}" transform="translate(${x},${y})">
          <title>${escapeHtml(node.name)}</title>${halo}
          <rect class="node-body" x="-59" y="-29" width="118" height="58" rx="${node.type === "decision" ? 7 : 13}"
            fill="${fill}" stroke="${color}" stroke-width="${selected ? 1.8 : 1}"/>
          <text x="-44" y="-6" class="node-symbol" fill="${ink}">${symbol}</text>
          <text x="-27" y="-6" class="node-caption">${caption}</text>
          <text x="0" y="14" text-anchor="middle" class="node-label">${escapeHtml(label)}</text>
        </g>`;
    })
    .join("");
  element.innerHTML = `
    <div class="graph-tools" role="group" aria-label="Diagram zoom">
      <button class="zoom-button" data-zoom="out" aria-label="Zoom out" title="Zoom out">−</button>
      <button class="zoom-button fit" data-zoom="fit" aria-label="Fit the diagram to the view" title="Fit to view">Fit</button>
      <button class="zoom-button" data-zoom="in" aria-label="Zoom in" title="Zoom in">+</button>
    </div>
    <div class="graph-scroll">
      <svg class="network-svg" viewBox="0 0 ${width} ${height}" data-width="${width}" data-height="${height}"
        aria-label="Directed graph of ${escapeHtml(model.name)}">
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6" fill="none" stroke="#b9b1c9" stroke-width="1"/>
          </marker>
        </defs>
        ${edges}${nodeMarkup}
      </svg>
    </div>
    <span class="graph-label">Click a node to inspect its values · Dashed arrows indicate observations<span
      class="graph-hint" hidden> · Drag or scroll to see the whole graph; Ctrl/⌘ + scroll zooms</span></span>`;
  applyZoom();
}

function currentZoom(scroll, svg) {
  if (state.graphZoom) return state.graphZoom;
  // Automatic: natural size when it fits, otherwise shrink to fit (but keep labels readable).
  return Math.max(0.55, Math.min(1, (scroll.clientWidth - 4) / Number(svg.dataset.width)));
}

function applyZoom(anchor) {
  const scroll = document.querySelector(".graph-scroll");
  const svg = scroll?.querySelector("svg");
  if (!svg) return;
  const before = anchor && {
    x: (scroll.scrollLeft + anchor.x) / Math.max(1, scroll.scrollWidth),
    y: (scroll.scrollTop + anchor.y) / Math.max(1, scroll.scrollHeight),
  };
  const zoom = currentZoom(scroll, svg);
  svg.setAttribute("width", Math.round(Number(svg.dataset.width) * zoom));
  svg.setAttribute("height", Math.round(Number(svg.dataset.height) * zoom));
  if (before) {
    scroll.scrollLeft = before.x * scroll.scrollWidth - anchor.x;
    scroll.scrollTop = before.y * scroll.scrollHeight - anchor.y;
  }
  scroll.classList.toggle(
    "pannable",
    scroll.scrollWidth > scroll.clientWidth + 1 || scroll.scrollHeight > scroll.clientHeight + 1,
  );
  updateGraphHint();
}

function zoomGraph(kind, anchor) {
  const scroll = document.querySelector(".graph-scroll");
  const svg = scroll?.querySelector("svg");
  if (!svg) return;
  const zoom = currentZoom(scroll, svg);
  if (kind === "fit") {
    const fit = Math.min(
      (scroll.clientWidth - 4) / Number(svg.dataset.width),
      460 / Number(svg.dataset.height),
      1.5,
    );
    state.graphZoom = Math.max(ZOOM_MIN, fit);
  } else {
    const factor = typeof kind === "number" ? kind : kind === "in" ? 1.25 : 1 / 1.25;
    state.graphZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * factor));
  }
  applyZoom(anchor || { x: scroll.clientWidth / 2, y: scroll.clientHeight / 2 });
}

// Keep chart labels about 12px on screen however wide the chart is drawn.
function fitChartText() {
  for (const svg of document.querySelectorAll(".sensitivity-svg")) {
    const scale = svg.getScreenCTM()?.a; // screen pixels per chart unit, after letterboxing
    if (scale) svg.style.fontSize = `${12 / scale}px`;
  }
}

// Mention scrolling only when the graph is larger than its card.
function updateGraphHint() {
  const scroll = document.querySelector(".graph-scroll");
  const hint = document.querySelector(".graph-hint");
  if (scroll && hint)
    hint.hidden = scroll.scrollWidth <= scroll.clientWidth + 1 && scroll.scrollHeight <= scroll.clientHeight + 1;
}

function renderInspector() {
  const inspector = document.getElementById("inspector");
  if (!inspector) return;
  const node = graphNodes().find((n) => n.name === state.selected);
  if (!node) {
    inspector.innerHTML = '<p class="graph-empty">Add a node to see its details here.</p>';
    return;
  }
  let header;
  let rows;
  // Large tables are summarised here; the full table is in Model data.
  const INSPECTOR_ROWS = 160;
  let total;
  if (node.type === "chance") {
    header = `<th>${node.parents.length ? "Parent states / outcome" : "State"}</th><th>Probability</th>`;
    total = node.cpt.length * node.states.length;
    rows = node.cpt.slice(0, Math.ceil(INSPECTOR_ROWS / node.states.length)).flatMap((row) =>
      node.states.map((value) => {
        const when = node.parents.length ? `<small>${escapeHtml(assignmentText(row.when))}</small><br>` : "";
        return `<tr><td>${when}${escapeHtml(value)}</td><td>${pct(row.probabilities[value])}%</td></tr>`;
      }),
    );
  } else if (node.type === "utility") {
    header = "<th>State / action</th><th>Utility</th>";
    total = node.table.length;
    rows = node.table
      .slice(0, INSPECTOR_ROWS)
      .map((row) => `<tr><td>${escapeHtml(assignmentText(row.when))}</td><td>${num(row.value)}</td></tr>`);
  } else {
    header = "<th>Available actions</th><th></th>";
    total = node.states.length;
    rows = node.states.map((value) => `<tr><td>${escapeHtml(value)}</td><td>→</td></tr>`);
  }
  if (total > rows.length)
    rows.push(`<tr><td colspan="2"><small>${(total - rows.length).toLocaleString()} more rows in Model data</small></td></tr>`);
  const table = `<table class="mini-table"><thead><tr>${header}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
  const description =
    node.type === "chance"
      ? node.parents.length
        ? "Conditional probability table"
        : "Prior probability distribution"
      : node.type === "utility"
        ? "Preference scores for each outcome"
        : `Observes: ${escapeHtml((node.information || []).join(", ") || "no variables")}`;
  inspector.innerHTML = `
    <div><div class="node-type">${node.type} node</div><h3>${escapeHtml(node.name)}</h3><p>${description}</p></div>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Table for ${escapeHtml(node.name)}">${table}</div>`;
}

function assignmentEditor(kind, values, allowed, canAdd = true) {
  const entries = Object.entries(values);
  const options = state.model.chance_nodes.filter((node) => allowed.includes(node.name));
  const rows = entries.map(([name, value], index) => {
    const node = options.find((item) => item.name === name);
    const variables = options
      .filter((item) => item.name === name || !has(values, item.name))
      .map((item) => {
        const selected = item.name === name ? "selected" : "";
        return `<option value="${escapeHtml(item.name)}" ${selected}>${escapeHtml(item.name)}</option>`;
      })
      .join("");
    const states = node.states
      .map((option) => {
        const selected = option === value ? "selected" : "";
        return `<option value="${escapeHtml(option)}" ${selected}>${escapeHtml(option)}</option>`;
      })
      .join("");
    const remove =
      kind === "event" && entries.length === 1
        ? ""
        : `<button class="remove-row" data-remove="${kind}" data-name="${escapeHtml(name)}"` +
          ` aria-label="Remove ${escapeHtml(name)}">×</button>`;
    return `
      <div class="assignment-row">
        <select aria-label="${kind} variable ${index + 1}" data-kind="${kind}" data-name="${escapeHtml(name)}"
          data-field="variable">${variables}</select>
        <span class="equals">=</span>
        <select aria-label="${kind} state ${index + 1}" data-kind="${kind}" data-name="${escapeHtml(name)}"
          data-field="state">${states}</select>
        ${remove}
      </div>`;
  });
  const noun = kind === "event" ? "event" : "observation";
  const add =
    canAdd && options.some((node) => !has(values, node.name))
      ? `<button class="button text add-row" data-add="${kind}">+ Add ${noun}</button>`
      : "";
  return `<div class="assignment-list">${rows.join("")}</div>${add}`;
}

function renderWorkbench() {
  const root = document.getElementById("workbench");
  if (!root) return;
  saveSession();
  if (state.tab === "model") {
    renderEditor(root);
    return;
  }
  const decision = state.model.decision_nodes?.[0];
  if (state.tab === "utility" && !decision) {
    root.innerHTML = `
      <div class="empty-state">
        ${icon("diamond")}
        <h3>Give your network a decision.</h3>
        <p>This example contains chance variables only. Open the umbrella example to compare actions,
          or add a decision node and a utility table in Model data.</p>
        <button class="button primary" data-example="umbrella">Explore the umbrella decision ${icon("arrow")}</button>
      </div>`;
    return;
  }
  const names = state.model.chance_nodes.map((node) => node.name);
  let controls;
  if (state.tab === "probability") {
    const modes = ["marginal", "joint", "conditional"]
      .map((mode) => {
        const active = state.mode === mode;
        const label = mode[0].toUpperCase() + mode.slice(1);
        return `<button data-mode="${mode}" class="${active ? "active" : ""}"
          aria-pressed="${active}">${label}</button>`;
      })
      .join("");
    const evidenceHint = Object.keys(state.evidence).length
      ? ""
      : '<p class="field-hint">Add an observation to condition this probability.</p>';
    const modeHint =
      state.mode === "joint"
        ? "Combine two or more events. Unselected variables are summed out."
        : "Sum over the other variables to find this event’s probability.";
    const evidenceBlock =
      state.mode === "conditional"
        ? `<div class="field-group">
            <div class="form-label">Evidence<span>What have you observed?</span></div>
            ${assignmentEditor("evidence", state.evidence, names)}${evidenceHint}
          </div>`
        : `<p class="field-hint">${modeHint}</p>`;
    let actionBlock = "";
    if (actionAffectsChance()) {
      const actions = decision.states
        .map((action) => {
          const selected = state.action === action ? "selected" : "";
          return `<option value="${escapeHtml(action)}" ${selected}>${escapeHtml(action)}</option>`;
        })
        .join("");
      actionBlock = `<div class="field-group">
          <label class="form-label" for="action-select">Fixed action<span>${escapeHtml(decision.name)}</span></label>
          <select id="action-select" class="action-select" data-action-select>${actions}</select>
          <p class="field-hint">The selected action is set for this calculation.</p>
        </div>`;
    }
    const eventLabel = state.mode === "joint" ? "Joint event" : "Event";
    controls = `<div class="section-caption"><span class="step">01</span><h3>Build your query</h3></div>
      <div class="segmented" aria-label="Probability type">${modes}</div>
      <div class="form-label">${eventLabel}<span>What do you want to know?</span></div>
      ${assignmentEditor("event", state.event, names, state.mode !== "marginal")}
      ${evidenceBlock}
      ${actionBlock}`;
  } else {
    const information = decision.information || [];
    const hint = information.length
      ? "Only observations available before this decision can be used. " +
        "Unobserved information is evaluated in the policy below."
      : "This decision has no information parents; compare actions using prior probabilities.";
    controls = `<div class="section-caption"><span class="step">01</span><h3>Find the best decision</h3></div>
      <p class="utility-intro">Compare every action for <strong>${escapeHtml(decision.name)}</strong>.
        Each outcome’s utility is weighted by its probability.</p>
      <div class="utility-formula">EU(a | e) = Σ P(x | e, a) U(x, a)<br>MEU(e) = max<sub>a</sub> EU(a | e)</div>
      <div class="field-group">
        <div class="form-label">Available evidence<span>Optional observations</span></div>
        ${assignmentEditor("decisionEvidence", state.decisionEvidence, information)}
        <p class="field-hint">${hint}</p>
      </div>`;
  }
  const methodOption = (value, label) =>
    `<option value="${value}" ${state.method === value ? "selected" : ""}>${label}</option>`;
  controls += `
      <div class="field-group">
        <label class="form-label" for="method-select">Inference algorithm</label>
        <select id="method-select" class="action-select">
          ${methodOption("elimination", "Variable elimination")}
          ${methodOption("enumeration", "Enumeration · all possible worlds")}
        </select>
      </div>`;
  if (state.tab === "utility") controls += utilityEditor();
  const idle = state.tab === "probability" ? "Calculate probability" : "Evaluate decisions";
  const label = state.busy ? "Calculating…" : idle;
  const disabled = state.busy ? "disabled" : "";
  const error = state.error ? `<div class="error-message" role="alert">${escapeHtml(state.error)}</div>` : "";
  const result = state.tab === "probability" ? probabilityResult() : utilityResult();
  root.innerHTML = `
    <div class="workbench">
      <div class="controls">
        ${controls}
        <button class="button primary calculate" data-command="calculate" ${disabled}>${label}${icon("arrow")}</button>
      </div>
      <div class="result-panel" aria-live="polite">${error}${result}</div>
    </div>`;
  fitChartText();
}

function probabilityResult() {
  const result = state.query;
  if (!result) {
    const heading = state.busy ? "Following the probabilities…" : "A question worth calculating.";
    return `
      <div class="empty-state">
        ${icon("chart")}
        <h3>${heading}</h3>
        <p>Choose your event and observations, then calculate to see the exact probability and the inference steps.</p>
      </div>`;
  }
  const hasEvidence = Object.keys(result.evidence).length > 0;
  const action =
    result.action !== null
      ? `${hasEvidence ? ", " : " | "}do(${state.model.decision_nodes[0].name} = ${result.action})`
      : "";
  const conditions = hasEvidence ? " | " + assignmentText(result.evidence) : "";
  const formula = `P(${assignmentText(result.event)}${conditions}${action})`;
  const distribution = result.distribution.length
    ? result.distribution
    : [
        { state: "Selected event", probability: result.probability },
        { state: "All other outcomes", probability: 1 - result.probability },
      ];
  const target = Object.values(result.event)[0];
  const bars = distribution
    .map((row, index) => {
      const selected = result.distribution.length ? row.state === target : index === 0;
      return `
        <div>
          <div class="bar-label"><span>${escapeHtml(row.state)}</span><strong>${pct(row.probability)}%</strong></div>
          <div class="bar-track">
            <div class="bar-fill ${selected ? "selected" : ""}" style="width:${row.probability * 100}%"></div>
          </div>
        </div>`;
    })
    .join("");
  const steps = result.trace
    .map((step) => {
      const scope = step.variable
        ? ` Remaining scope: ${escapeHtml(step.scope.join(", ") || "scalar")}.`
        : "";
      return `<li>${escapeHtml(step.description)}${scope}</li>`;
    })
    .join("");
  const given = hasEvidence ? "given the observed evidence" : "under this model";
  const evidenceProbability = escapeHtml(
    probabilityText(result.evidence_probability, result.log10_evidence_probability),
  );
  return `
    <div class="result-topline">
      <span class="result-eyebrow">02 / Probability result</span>
      <span class="success-label">✓ ${escapeHtml(result.method)}</span>
    </div>
    <div class="formula">${escapeHtml(formula)}</div>
    <div class="big-result">${pct(result.probability)}<span>%</span></div>
    <p class="result-explainer">${num(result.probability)} probability ${given}.</p>
    <div class="distribution">${bars}</div>
    <details class="trace">
      <summary>How was this calculated?</summary>
      <ol>
        ${steps}
        <li>Evidence probability: <span class="numeric">${evidenceProbability}</span>.</li>
      </ol>
    </details>
    ${worldsPanel()}`;
}

function utilityResult() {
  const result = state.utility;
  if (!result) {
    const heading = state.busy ? "Weighing the outcomes…" : "Turn probabilities into a decision.";
    return `
      <div class="empty-state">
        ${icon("diamond")}
        <h3>${heading}</h3>
        <p>Evaluate the available actions to find their expected utilities and the optimal observation policy.</p>
      </div>`;
  }
  const optimal = (action) => result.optimal_actions.includes(action);
  const actionRows = result.actions
    .map(
      (row) => `
        <tr class="${optimal(row.action) ? "best" : ""}">
          <td>${escapeHtml(row.action)}</td><td>${num(row.expected_utility)}</td>
          <td>${optimal(row.action) ? '<span class="optimal-badge">Optimal</span>' : "—"}</td>
        </tr>`,
    )
    .join("");
  const policyHeader = result.actions.map((a) => `<th>EU(${escapeHtml(a.action)})</th>`).join("");
  const policyRows = result.policy
    .map((row) => {
      const values = result.actions
        .map((a) => {
          const value = row.actions
            ? num(row.actions.find((r) => r.action === a.action).expected_utility)
            : "—";
          return `<td>${value}</td>`;
        })
        .join("");
      const best = row.action === null ? "Unreachable" : escapeHtml(row.optimal_actions.join(" / "));
      return `<tr><td>${escapeHtml(assignmentText(row.when) || "None")}</td>${values}<td>${best}</td></tr>`;
    })
    .join("");
  const unobserved = result.unobserved_information;
  const policyIntro = unobserved.length
    ? `If you can observe ${escapeHtml(unobserved.join(", "))} before acting, adapt your choice:`
    : "Your available observations are included in the decision above.";
  const fullPolicy = Object.keys(result.evidence).length
    ? '<button class="button text" data-command="full-policy">Compare every possible forecast / observation →</button>'
    : "";
  const netMetric = result.observation_cost
    ? `<span>Net of observation costs<strong>${result.net_value_of_information >= 0 ? "+" : ""}${num(
        result.net_value_of_information,
      )}</strong></span>`
    : "";
  return `
    <div class="result-topline">
      <span class="result-eyebrow">02 / Maximum expected utility</span>
      <span class="success-label">✓ All actions evaluated</span>
    </div>
    <div class="formula">MEU(${escapeHtml(assignmentText(result.evidence) || "current information")})</div>
    <div class="utility-score">
      <div class="big-result">${num(result.maximum_expected_utility)}</div><span class="units">utility units</span>
    </div>
    <p class="best-action">${result.optimal_actions.length > 1 ? "Equally good actions" : "Best action now"}:
      <strong>${escapeHtml(result.optimal_actions.join(" or "))}</strong></p>
    <table class="action-table">
      <thead><tr><th>Action</th><th>Expected utility</th><th>Result</th></tr></thead>
      <tbody>${actionRows}</tbody>
    </table>
    ${sensitivityChart()}
    <div class="policy-box">
      <h3>${unobserved.length ? "A policy for every observation" : "Optimal policy"}</h3>
      <p>${policyIntro}</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Policy table">
        <table class="policy-table">
          <thead><tr><th>Observation</th>${policyHeader}<th>Best action</th></tr></thead>
          <tbody>${policyRows}</tbody>
        </table>
      </div>
      ${fullPolicy}
      <div class="policy-metrics">
        <span>Expected policy utility<strong>${num(result.policy_expected_utility)}</strong></span>
        <span>Value of remaining information<strong>+${num(result.value_of_information)}</strong></span>
        ${netMetric}
      </div>
    </div>
    ${informationValues(result)}`;
}

function informationValues(result) {
  const rows = result.information_values || [];
  if (!rows.length) return "";
  const verdict = (row) =>
    row.worth_observing
      ? '<span class="optimal-badge">Worth observing</span>'
      : row.value > 0
        ? '<span class="muted-badge">Costs more than it is worth</span>'
        : '<span class="muted-badge">Would not change the decision</span>';
  const body = rows
    .map(
      (row) => `<tr><td>${escapeHtml(row.variable)}</td><td>+${num(row.value)}</td><td>${num(row.cost)}</td>
        <td>${row.net_value >= 0 ? "+" : ""}${num(row.net_value)}</td><td>${verdict(row)}</td></tr>`,
    )
    .join("");
  return `
    <div class="policy-box voi-box">
      <h3>Is it worth observing?</h3>
      <p>The value of observing each variable on its own before deciding: how much the best expected utility rises
        on average. Set observation costs under <strong>Edit your preferences</strong>.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Value of information table">
        <table class="policy-table voi-table">
          <thead><tr><th>Observe</th><th>Value</th><th>Cost</th><th>Net value</th><th>Verdict</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>`;
}

function actionAffectsChance() {
  const name = state.model.decision_nodes?.[0]?.name;
  return Boolean(name && state.model.chance_nodes.some((node) => node.parents.includes(name)));
}

function utilityEditor() {
  const tables = state.utilityDraft
    .map((node, ni) => {
      const inputs = node.table
        .map((row, ri) => {
          const when = assignmentText(row.when);
          return `
            <label>
              <span>${escapeHtml(when || "Constant utility")}</span>
              <input type="number" step="any" value="${escapeHtml(row.value)}" data-utility="${ni}" data-row="${ri}"
                aria-label="${escapeHtml(node.name + ": " + when)}">
            </label>`;
        })
        .join("");
      return `<div class="utility-inputs"><h4>${escapeHtml(node.name)}</h4>${inputs}</div>`;
    })
    .join("");
  const information = state.model.decision_nodes?.[0]?.information || [];
  const costs = information.length
    ? `<div class="utility-inputs"><h4>Cost of observing</h4>${information
        .map(
          (name) => `
            <label>
              <span>${escapeHtml(name)}</span>
              <input type="number" step="any" min="0" placeholder="0" value="${escapeHtml(has(state.costDraft, name) ? state.costDraft[name] : "")}"
                data-cost="${escapeHtml(name)}" aria-label="Cost of observing ${escapeHtml(name)}">
            </label>`,
        )
        .join("")}</div>`
    : "";
  return `
      <details class="utility-editor" open>
        <summary>Edit your preferences</summary>
        <p class="field-hint">Utility units reflect how much each outcome matters to you.
          Changes apply when you evaluate.</p>
        ${tables}${costs}
      </details>`;
}

function sensitivityTarget() {
  const decision = state.model.decision_nodes?.[0];
  if (!decision || actionAffectsChance()) return null;
  return state.model.chance_nodes.find(
    (node) =>
      node.states.length === 2 &&
      state.model.utility_nodes.every((u) => u.parents.every((p) => p === node.name || p === decision.name)),
  )?.name;
}

const LINE_COLORS = ["#7652c7", "#34967e", "#d07a2c", "#3f7fd0", "#c24f74", "#8a8f2a", "#5e6b7d", "#a7479c"];

function sensitivityMessage(result) {
  if (result.always_tied) return "All actions tie at every probability.";
  const regions = result.regions || [];
  if (regions.length <= 1) {
    const best = regions[0]?.optimal_actions || [];
    return best.length > 1
      ? `${best.join(" and ")} tie for best across the full probability range.`
      : `${best[0] ?? "One action"} stays best across the full probability range.`;
  }
  if (result.lines.length === 2 && regions.length === 2)
    return `The actions tie at ${pct(regions[1].from)}% probability: ${regions[0].optimal_actions.join(" / ")} below, ${regions[1].optimal_actions.join(" / ")} above.`;
  return (
    "Best action by probability: " +
    regions
      .map((region) => `${region.optimal_actions.join(" / ")} from ${pct(region.from)}% to ${pct(region.to)}%`)
      .join("; ") +
    "."
  );
}

function sensitivityChart() {
  const result = state.sensitivity;
  if (!result) return "";
  const values = result.lines.flatMap((line) => [line.intercept, line.intercept + line.slope]);
  const min = Math.min(...values, 0),
    max = Math.max(...values, 0);
  const extent = Math.max(1, max - min);
  const x = (p) => 46 + 364 * p,
    y = (value) => 155 - ((value - min) / extent) * 125;
  const thresholds = result.thresholds || (result.threshold === null ? [] : [result.threshold]);
  const message = sensitivityMessage(result);
  const outcome = `${escapeHtml(result.target)} = ${escapeHtml(result.state)}`;
  const description =
    `Expected utility versus probability of ${escapeHtml(result.target)} equals ${escapeHtml(result.state)}. ` +
    escapeHtml(message);
  const gridLines = [min, min + extent / 2, min + extent]
    .map(
      (v) =>
        `<path d="M46 ${y(v)}H410" stroke="#eee8f5"/>` +
        `<text x="37" y="${y(v) + 4}" text-anchor="end">${num(v)}</text>`,
    )
    .join("");
  const crossovers = thresholds
    .map((t) => `<path d="M${x(t)} 21V155" stroke="#a79bb8" stroke-dasharray="4 4"/>`)
    .join("");
  const lines = result.lines
    .map((line, i) => {
      const color = LINE_COLORS[i % LINE_COLORS.length];
      const dot = y(line.intercept + line.slope * result.posterior);
      return (
        `<path d="M46 ${y(line.intercept)}L410 ${y(line.intercept + line.slope)}" stroke="${color}"` +
        ` stroke-width="2.5" fill="none"/><circle cx="${x(result.posterior)}" cy="${dot}" r="4" fill="${color}"/>`
      );
    })
    .join("");
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    .map((p) => `<text x="${x(p)}" y="174" text-anchor="middle">${Math.round(p * 100)}%</text>`)
    .join("");
  const legend = result.lines
    .map((line, i) => {
      const sign = line.intercept < 0 ? "−" : "+";
      const equation = `${num(line.slope)}p ${sign} ${num(Math.abs(line.intercept))}`;
      return `<span><i style="background:${LINE_COLORS[i % LINE_COLORS.length]}"></i>${escapeHtml(line.action)}: ${equation}</span>`;
    })
    .join("");
  return `<div class="sensitivity">
    <h3>When does the best action change?</h3>
    <p>${escapeHtml(message)} Current P(${outcome}) is <strong>${pct(result.posterior)}%</strong>.</p>
    <svg class="sensitivity-svg" viewBox="0 0 440 204" role="img" aria-label="${description}">
      ${gridLines}
      ${crossovers}
      <path d="M${x(result.posterior)} 21V155" stroke="#c2b3db" stroke-width="2"/>
      ${lines}
      ${ticks}
      <text x="228" y="196" text-anchor="middle">Probability of ${outcome}</text>
    </svg>
    <div class="sensitivity-legend">${legend}</div>
    <p class="field-hint">The vertical solid line marks your current belief; dashed lines mark where the best action
      changes. Edit utilities to move them. This chart varies belief without changing the saved probability tables.</p></div>`;
}

function worldsPanel() {
  if (!state.worlds) {
    return (
      '<div class="worlds-panel">' +
      '<button class="button text" data-command="worlds">Inspect all possible worlds →</button></div>'
    );
  }
  const evidence = state.mode === "conditional" ? state.evidence : {};
  const matches = (world, values) => Object.entries(values).every(([n, s]) => world[n] === s);
  const nodes = state.model.chance_nodes;
  const rows = state.worlds
    .map((row) => {
      const highlight = matches(row.assignment, evidence) && matches(row.assignment, state.event);
      const cells = nodes.map((n) => `<td>${escapeHtml(row.assignment[n.name])}</td>`).join("");
      return `<tr class="${highlight ? "matching" : ""}">${cells}<td>${num(row.probability)}</td></tr>`;
    })
    .join("");
  const total = state.worlds.reduce((sum, row) => sum + row.probability, 0);
  return `
    <details class="trace" open>
      <summary>Full joint table · ${state.worlds.length} worlds</summary>
      <p class="field-hint">Highlighted rows match both the event and evidence.
        Every hidden variable is included in the sum.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Full joint table">
        <table class="worlds-table">
          <thead><tr>${nodes.map((n) => `<th>${escapeHtml(n.name)}</th>`).join("")}<th>P(world)</th></tr></thead>
          <tbody>${rows}</tbody>
          <tfoot><tr><th colspan="${nodes.length}">Total probability</th><th>${num(total)}</th></tr></tfoot>
        </table>
      </div>
    </details>`;
}

// ---------- Model data: visual builder and JSON editor ----------

function ensureBuilder() {
  if (state.builder) return true;
  try {
    state.builder = Builder.normalize(structuredClone(state.model));
    state.builderDirty = false;
    return true;
  } catch (error) {
    state.error = error.message;
    state.editorView = "json";
    return false;
  }
}

function appliedJson() {
  return JSON.stringify(state.model, null, 2);
}

function renderEditor(root) {
  if (state.editorView === "builder") ensureBuilder();
  const error = state.error ? `<div class="error-message" role="alert">${escapeHtml(state.error)}</div>` : "";
  const builder = state.editorView === "builder";
  const changed = builder ? state.builderDirty : state.draft !== appliedJson();
  const toggle = ["builder", "json"]
    .map((view) => {
      const active = state.editorView === view;
      return `<button data-view="${view}" class="${active ? "active" : ""}" aria-pressed="${active}">
        ${view === "builder" ? "Visual builder" : "JSON"}</button>`;
    })
    .join("");
  const intro = builder
    ? "Add variables, choose what each depends on, and fill in its table. The diagram above previews your draft."
    : "Edit the network’s states, dependencies, probability tables, and utilities as JSON.";
  const body = builder
    ? Builder.render(state.builder, state.selected)
    : `<textarea id="model-editor" aria-label="Network model JSON" spellcheck="false"
        data-focus="json-editor">${escapeHtml(state.draft)}</textarea>
      <p class="help-box">Each CPT row must sum to 1 and cover one parent assignment. A root node uses
        <code>"when": {}</code>. Decision <code>information</code> lists the chance variables observed before
        acting, and optional <code>observation_costs</code> give what observing each one costs. Utility tables
        contain numeric preference scores. Supports one decision and multiple additive utilities.</p>`;
  root.innerHTML = `
    <div class="editor">
      <div class="editor-top">
        <div>
          <div class="segmented editor-switch" role="group" aria-label="Editor view">${toggle}</div>
          <p>${intro} Apply your changes to validate the model and update the workspace.</p>
        </div>
        <div class="editor-actions">
          <button class="button secondary" data-command="import">${icon("upload")} Import</button>
          <button class="button secondary" data-command="discard" ${changed ? "" : "disabled"}>Discard changes</button>
          <button class="button primary" data-command="apply" data-focus="apply"
            ${state.busy ? "disabled" : ""}>${icon("check")} Validate & apply</button>
        </div>
      </div>
      ${error}
      ${body}
      <div class="editor-status" role="status">${escapeHtml(state.editorMessage)}</div>
    </div>`;
}

function refreshBuilder(focusKey = document.activeElement?.dataset?.focus) {
  renderGraphCard();
  renderWorkbench();
  restoreFocus(focusKey);
}

function markBuilderChanged() {
  const first = !state.builderDirty;
  state.builderDirty = true;
  state.editorMessage = "";
  if (first) renderGraphCard();
  saveSession(300);
}

// Open a model in the builder, saying which missing table rows were filled with defaults.
function loadIntoBuilder(raw) {
  const report = {};
  state.builder = Builder.normalize(raw, report);
  const filled = Object.entries(report);
  if (filled.length)
    state.editorMessage =
      "Some table rows were missing and were filled with defaults (uniform probabilities, utility 0): " +
      filled.map(([name, count]) => `${name} (${count} row${count === 1 ? "" : "s"})`).join(", ") +
      ". They are highlighted below; review them before you apply.";
  const names = Builder.allNames(state.builder);
  if (!names.includes(state.selected)) state.selected = (filled[0]?.[0] ?? names[0]) || null;
}

function switchEditorView(view) {
  if (view === state.editorView) return;
  state.error = "";
  if (view === "json") {
    if (state.builder && state.builderDirty) state.draft = JSON.stringify(state.builder, null, 2);
    state.editorView = "json";
  } else {
    try {
      loadIntoBuilder(JSON.parse(state.draft));
      state.builderDirty = state.draft !== appliedJson();
      state.editorView = "builder";
    } catch (error) {
      state.error =
        error instanceof SyntaxError
          ? `The JSON is not valid, so the builder cannot open it: ${error.message}`
          : `The builder cannot open this model: ${error.message}`;
    }
  }
  refreshBuilder();
}

function builderAction(target) {
  const model = state.builder;
  const name = target.dataset.name;
  let focus = target.dataset.focus;
  try {
    switch (target.dataset.b) {
      case "select":
        state.selected = name;
        break;
      case "add-chance":
        state.selected = Builder.addChance(model);
        focus = "node-name";
        break;
      case "add-decision":
        state.selected = Builder.addDecision(model);
        focus = "node-name";
        break;
      case "add-utility":
        state.selected = Builder.addUtility(model);
        focus = "node-name";
        break;
      case "delete":
        Builder.removeNode(model, name);
        state.selected = Builder.allNames(model)[0] ?? null;
        focus = state.selected ? `node:${state.selected}` : "model-name";
        toast(`Deleted ${name}. Discard changes to undo.`);
        break;
      case "add-state":
        Builder.addState(model, name);
        break;
      case "remove-state":
        Builder.removeState(model, name, target.dataset.state);
        focus = `add-state:${name}`;
        break;
      case "normalize-row":
        Builder.normalizeTable(model, name, Number(target.dataset.row));
        break;
      case "normalize-all":
        Builder.normalizeTable(model, name);
        break;
    }
    if (target.dataset.b !== "select") markBuilderChanged();
    state.error = "";
  } catch (error) {
    state.error = error.message;
  }
  refreshBuilder(focus);
}

// A field's change event fires when it loses focus, which for a mouse click happens on mousedown,
// before the click lands. Re-rendering at that moment would replace the button under the pointer and
// swallow the click, so while a pointer is pressed the refresh waits until it is released.
let pointerPressed = false;
let refreshPending = false;
document.addEventListener("pointerdown", () => (pointerPressed = true), true);
const releasePointer = () => {
  pointerPressed = false;
  if (refreshPending) {
    refreshPending = false;
    setTimeout(() => refreshBuilder(), 0); // runs after the click has been handled
  }
};
document.addEventListener("pointerup", releasePointer, true);
document.addEventListener("pointercancel", releasePointer, true);

function scheduleBuilderRefresh() {
  if (pointerPressed) refreshPending = true;
  // After Tab or Enter, re-render once the browser has moved focus, then put it back.
  else setTimeout(() => refreshBuilder(), 0);
}

function builderChange(target) {
  const model = state.builder;
  const name = target.dataset.name;
  try {
    switch (target.dataset.bfield) {
      case "model-name":
        model.name = target.value.trim().slice(0, 80) || "Untitled network";
        break;
      case "node-name":
        state.selected = Builder.renameNode(model, name, target.value);
        break;
      case "state-name":
        Builder.renameState(model, name, target.dataset.state, target.value);
        break;
      case "parent": {
        const parents = new Set(Builder.find(model, name).node.parents);
        if (target.checked) parents.add(target.dataset.parent);
        else parents.delete(target.dataset.parent);
        Builder.setParents(model, name, [...parents]);
        break;
      }
      case "information": {
        const items = new Set(model.decision_nodes[0].information);
        if (target.checked) items.add(target.dataset.parent);
        else items.delete(target.dataset.parent);
        Builder.setInformation(model, [...items]);
        break;
      }
    }
    markBuilderChanged();
    state.error = "";
  } catch (error) {
    state.error = error.message;
  }
  scheduleBuilderRefresh();
}

function builderInput(input) {
  const model = state.builder;
  if (!model) return;
  if (input.dataset.bprob !== undefined) {
    const sum = Builder.setProbability(
      model,
      input.dataset.name,
      Number(input.dataset.row),
      input.dataset.state,
      input.value,
    );
    const cell = app.querySelector(`[data-sum-row="${input.dataset.row}"]`);
    if (cell) {
      cell.textContent = Builder.formatSum(sum);
      cell.className = `row-sum ${Builder.sumClass(sum)}`;
    }
  } else if (input.dataset.butil !== undefined) {
    Builder.setUtility(model, input.dataset.name, Number(input.dataset.row), input.value);
  } else if (input.dataset.bcost !== undefined) {
    Builder.setCost(model, input.dataset.name, input.value);
  } else return;
  markBuilderChanged();
}

function discardDraft() {
  state.builder = null;
  state.builderDirty = false;
  state.draft = appliedJson();
  state.error = "";
  state.editorMessage = "Changes discarded. The editor shows the applied model again.";
  refreshBuilder("apply");
}

function invalidate() {
  state.request++;
  state.query = null;
  state.utility = null;
  state.sensitivity = null;
  state.worlds = null;
  state.busy = false;
  state.error = "";
  renderWorkbench();
}

function preferencesChanged() {
  state.request++;
  state.busy = false;
  state.utility = null;
  state.sensitivity = null;
  document.querySelector(".result-panel").innerHTML =
    '<div class="empty-state"><h3>Preferences changed.</h3>' +
    "<p>Select Evaluate decisions to apply these utilities and update the recommendation.</p></div>";
  const button = document.querySelector('[data-command="calculate"]');
  if (button) {
    button.disabled = false;
    button.textContent = "Evaluate decisions →";
  }
  saveSession(300);
}

function draftedModel() {
  const model = structuredClone(state.model);
  model.utility_nodes = state.utilityDraft.map((node) => ({
    ...node,
    table: node.table.map((row) => {
      if (String(row.value).trim() === "" || !Number.isFinite(Number(row.value)))
        throw new Error("Enter a finite number in every utility field.");
      return { ...row, value: Number(row.value) };
    }),
  }));
  const decision = model.decision_nodes[0];
  if (decision) {
    const costs = {};
    for (const name of decision.information || []) {
      const raw = String(has(state.costDraft, name) ? state.costDraft[name] : "").trim();
      if (raw === "") continue;
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0) throw new Error(`Enter a cost of zero or more for observing ${name}.`);
      if (value > 0) costs[name] = value;
    }
    if (Object.keys(costs).length) decision.observation_costs = costs;
    else delete decision.observation_costs;
  }
  return model;
}

// Copy evaluated utilities and observation costs into the unapplied builder draft wherever the
// draft still has the same table rows, so applying the draft later does not undo them.
function syncPreferencesIntoDraft(model) {
  const draft = state.builder;
  if (!draft) return;
  const key = (when) => JSON.stringify(Object.entries(when).sort());
  for (const node of model.utility_nodes) {
    const target = draft.utility_nodes.find((item) => item.name === node.name);
    if (!target) continue;
    const values = new Map(node.table.map((row) => [key(row.when), row.value]));
    for (const row of target.table) if (values.has(key(row.when))) row.value = values.get(key(row.when));
  }
  const decision = model.decision_nodes[0];
  const target = draft.decision_nodes[0];
  if (decision && target && decision.name === target.name) {
    const costs = decision.observation_costs || {};
    target.observation_costs = Object.fromEntries(
      target.information.filter((name) => has(costs, name)).map((name) => [name, costs[name]]),
    );
  }
}

async function calculate() {
  const tab = state.tab;
  if (tab === "model") return;
  const token = ++state.request;
  state.busy = true;
  state.error = "";
  if (tab === "probability") state.query = null;
  else state.utility = null;
  renderWorkbench();
  try {
    if (tab === "probability" && state.mode === "conditional" && !Object.keys(state.evidence).length)
      throw new Error("Add at least one observation for a conditional probability.");
    if (tab === "probability" && state.mode === "joint" && Object.keys(state.event).length < 2)
      throw new Error("Add at least two variables for a joint probability.");
    let model = state.model;
    if (tab === "utility") {
      model = draftedModel();
      await api("/api/validate", { model });
      if (token !== state.request) return;
    }
    const result = await api(tab === "probability" ? "/api/query" : "/api/utility", {
      model,
      event: state.event,
      evidence:
        tab === "probability" ? (state.mode === "conditional" ? state.evidence : {}) : state.decisionEvidence,
      action: actionAffectsChance() ? state.action : null,
      method: state.method,
    });
    if (token !== state.request) return;
    if (tab === "probability") state.query = result;
    else {
      state.utility = result;
      // Evaluated preferences become part of the applied model (and of exports).
      if (state.draft === appliedJson()) state.draft = JSON.stringify(model, null, 2);
      if (!state.builderDirty) state.builder = null;
      else syncPreferencesIntoDraft(model);
      state.model = model;
      renderInspector();
      const target = sensitivityTarget();
      state.sensitivity = null;
      if (target) {
        const sensitivity = await api("/api/sensitivity", {
          model,
          target,
          evidence: state.decisionEvidence,
        });
        if (token !== state.request) return;
        state.sensitivity = sensitivity;
      }
    }
  } catch (error) {
    if (token === state.request) state.error = error.message;
  } finally {
    if (token === state.request) {
      state.busy = false;
      renderWorkbench();
    }
  }
}

async function applyDraft() {
  let model;
  if (state.editorView === "builder" && state.builder) model = structuredClone(state.builder);
  else {
    try {
      model = JSON.parse(state.draft);
    } catch (error) {
      state.error = `The model is not valid JSON: ${error.message}`;
      renderWorkbench();
      return;
    }
  }
  const token = ++state.request;
  state.busy = true;
  state.error = "";
  renderWorkbench();
  try {
    await api("/api/validate", { model });
    if (token !== state.request) return;
    setModel(model);
    state.editorMessage =
      "✓ Model validated and applied. Open Probability inference or Decision & utility to explore it.";
    renderWorkbench();
    toast("Model validated and applied");
  } catch (error) {
    if (token === state.request) {
      state.busy = false;
      state.error = error.message;
      renderWorkbench();
    }
  }
}

// ---------- Saved sessions ----------

let saveTimer;
function saveSession(delay = 0) {
  clearTimeout(saveTimer);
  const write = () => {
    if (!state.model) return;
    const essentials = { version: 1, model: state.model, example: state.example, tab: state.tab };
    const full = {
      ...essentials,
      mode: state.mode,
      event: state.event,
      evidence: state.evidence,
      decisionEvidence: state.decisionEvidence,
      action: state.action,
      method: state.method,
      selected: state.selected,
      editorView: state.editorView,
      builder: state.builderDirty ? state.builder : null,
      builderDirty: state.builderDirty,
      draft: state.draft === appliedJson() ? null : state.draft,
      utilityDraft: state.utilityDraft,
      costDraft: state.costDraft,
    };
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(full));
    } catch {
      try {
        localStorage.setItem(SESSION_KEY, JSON.stringify(essentials)); // too large: keep the applied model only
      } catch {
        // Storage is unavailable (private window, blocked site data): the session is simply not saved.
      }
    }
  };
  if (delay) saveTimer = setTimeout(write, delay);
  else write();
}

function loadSession() {
  try {
    const session = JSON.parse(localStorage.getItem(SESSION_KEY));
    return session?.version === 1 && session.model ? session : null;
  } catch {
    return null;
  }
}

function clearSession() {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    // Nothing saved, nothing to clear.
  }
}

async function restoreSession(session) {
  Builder.normalize(session.model); // throws if the saved model is not even structurally a network
  const example = state.examples.some((item) => item.id === session.example) ? session.example : "custom";
  state.tab = session.tab === "utility" && !session.model.decision_nodes?.length ? "probability" : state.tab;
  setModel(session.model, example);
  const model = state.model;
  const states = new Map(model.chance_nodes.map((node) => [node.name, node.states]));
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const clean = (values, allowed = [...states.keys()]) =>
    Object.fromEntries(
      Object.entries(isObject(values) ? values : {}).filter(
        ([name, value]) => allowed.includes(name) && states.get(name)?.includes(value),
      ),
    );
  const decision = model.decision_nodes[0];
  const event = clean(session.event);
  if (Object.keys(event).length) state.event = event;
  state.evidence = clean(session.evidence);
  state.decisionEvidence = clean(session.decisionEvidence, decision?.information || []);
  if (["probability", "utility", "model"].includes(session.tab) && (session.tab !== "utility" || decision))
    state.tab = session.tab;
  if (["marginal", "joint", "conditional"].includes(session.mode)) state.mode = session.mode;
  if (["elimination", "enumeration"].includes(session.method)) state.method = session.method;
  if (decision?.states.includes(session.action)) state.action = session.action;
  if (["builder", "json"].includes(session.editorView)) state.editorView = session.editorView;
  if (typeof session.draft === "string") state.draft = session.draft;
  if (session.builder && session.builderDirty) {
    try {
      state.builder = Builder.normalize(session.builder);
      state.builderDirty = true;
    } catch {
      state.builder = null;
    }
  }
  const names = [...model.chance_nodes, ...model.decision_nodes, ...model.utility_nodes].map((node) => node.name);
  if (names.includes(session.selected) || (state.builder && Builder.allNames(state.builder).includes(session.selected)))
    state.selected = session.selected;
  const sameShape =
    Array.isArray(session.utilityDraft) &&
    session.utilityDraft.length === model.utility_nodes.length &&
    session.utilityDraft.every(
      (node, i) =>
        node?.name === model.utility_nodes[i].name && node.table?.length === model.utility_nodes[i].table.length,
    );
  if (sameShape) state.utilityDraft = session.utilityDraft;
  if (isObject(session.costDraft)) state.costDraft = { ...session.costDraft };
  render();
  if (state.tab !== "model") await calculate();
}

// ---------- Events ----------

app.addEventListener("click", async (event) => {
  const target = event.target.closest("button,[data-node-index]");
  if (!target) return;
  if (target.dataset.example) {
    await loadExample(target.dataset.example);
    return;
  }
  if (target.dataset.zoom) {
    zoomGraph(target.dataset.zoom);
    return;
  }
  if (target.dataset.b) {
    builderAction(target);
    return;
  }
  if (target.dataset.view) {
    switchEditorView(target.dataset.view);
    return;
  }
  if (target.dataset.nodeIndex !== undefined) {
    state.selected = graphNodes()[Number(target.dataset.nodeIndex)].name;
    renderGraph();
    renderInspector();
    if (state.tab === "model" && state.editorView === "builder") renderWorkbench();
    restoreFocus(`graph:${state.selected}`);
    return;
  }
  if (target.dataset.tab) {
    state.request++;
    state.busy = false;
    state.error = "";
    state.tab = target.dataset.tab;
    render();
    if (state.tab === "utility" && state.model.decision_nodes?.length && !state.utility) await calculate();
    return;
  }
  if (target.dataset.mode) {
    state.mode = target.dataset.mode;
    if (state.mode === "marginal") state.event = Object.fromEntries(Object.entries(state.event).slice(0, 1));
    if (state.mode === "joint" && Object.keys(state.event).length < 2) {
      const next = state.model.chance_nodes.find((node) => !has(state.event, node.name));
      if (next) state.event = { ...state.event, [next.name]: next.states[0] };
    }
    invalidate();
    return;
  }
  if (target.dataset.add) {
    const kind = target.dataset.add;
    const allowed =
      kind === "decisionEvidence"
        ? state.model.decision_nodes[0].information || []
        : state.model.chance_nodes.map((node) => node.name);
    const next = state.model.chance_nodes.find(
      (node) => allowed.includes(node.name) && !has(state[kind], node.name),
    );
    if (next) state[kind] = { ...state[kind], [next.name]: next.states[0] };
    invalidate();
    return;
  }
  if (target.dataset.remove) {
    delete state[target.dataset.remove][target.dataset.name];
    invalidate();
    return;
  }
  switch (target.dataset.command) {
    case "calculate":
      await calculate();
      break;
    case "worlds": {
      const token = state.request;
      target.disabled = true;
      try {
        const result = await api("/api/worlds", {
          model: state.model,
          action: actionAffectsChance() ? state.action : null,
        });
        if (token === state.request) {
          state.worlds = result.worlds;
          renderWorkbench();
        }
      } catch (error) {
        toast(error.message);
        target.disabled = false;
      }
      break;
    }
    case "full-policy":
      state.decisionEvidence = {};
      invalidate();
      await calculate();
      break;
    case "edit":
      state.tab = "model";
      state.error = "";
      state.request++;
      state.busy = false;
      render();
      break;
    case "apply":
      await applyDraft();
      break;
    case "discard":
      discardDraft();
      break;
    case "import":
      document.getElementById("file-input").click();
      break;
    case "export": {
      const blob = new Blob([JSON.stringify(state.model, null, 2) + "\n"], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = (state.model.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase() || "network") + ".json";
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      toast("Applied network exported as JSON");
      break;
    }
  }
});

app.addEventListener("change", (event) => {
  const target = event.target;
  if (target.dataset.bfield) {
    builderChange(target);
    return;
  }
  if (target.matches("[data-bprob], [data-butil], [data-bcost]")) {
    // Values were stored while typing; refresh the warnings in the node list.
    scheduleBuilderRefresh();
    return;
  }
  if (target.id === "method-select") {
    state.method = target.value;
    invalidate();
    return;
  }
  if (target.matches("[data-action-select]")) {
    state.action = target.value;
    invalidate();
    return;
  }
  if (!target.dataset.kind) return;
  const { kind, name, field } = target.dataset;
  if (field === "state") state[kind][name] = target.value;
  else {
    state[kind] = Object.fromEntries(
      Object.entries(state[kind]).map(([key, value]) =>
        key === name
          ? [target.value, state.model.chance_nodes.find((node) => node.name === target.value).states[0]]
          : [key, value],
      ),
    );
  }
  const value = target.value;
  invalidate();
  // Preserve keyboard focus after rebuilding the dependent selectors.
  const selector = [...app.querySelectorAll("select[data-kind]")].find(
    (item) => item.dataset.kind === kind && item.dataset.field === field && item.value === value,
  );
  selector?.focus();
});

app.addEventListener("input", (event) => {
  const input = event.target;
  if (input.id === "model-editor") {
    state.draft = input.value;
    const discard = app.querySelector('[data-command="discard"]');
    if (discard) discard.disabled = state.draft === appliedJson();
    saveSession(500);
    return;
  }
  if (input.matches("[data-bprob], [data-butil], [data-bcost]")) {
    builderInput(input);
    const discard = app.querySelector('[data-command="discard"]');
    if (discard) discard.disabled = false;
    return;
  }
  if (input.dataset.utility !== undefined) {
    state.utilityDraft[Number(input.dataset.utility)].table[Number(input.dataset.row)].value = input.value;
    preferencesChanged();
  } else if (input.dataset.cost !== undefined) {
    state.costDraft[input.dataset.cost] = input.value;
    preferencesChanged();
  }
});

// Drag the diagram to pan it (mouse only; touch screens scroll natively).
let pan = null;
app.addEventListener("pointerdown", (event) => {
  const scroll = event.target.closest(".graph-scroll");
  if (!scroll || event.pointerType !== "mouse" || event.button !== 0) return;
  if (event.target.closest(".node-group") || !scroll.classList.contains("pannable")) return;
  pan = {
    scroll,
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    left: scroll.scrollLeft,
    top: scroll.scrollTop,
  };
  scroll.setPointerCapture(event.pointerId);
  scroll.classList.add("panning");
});
app.addEventListener("pointermove", (event) => {
  if (!pan || event.pointerId !== pan.id) return;
  pan.scroll.scrollLeft = pan.left - (event.clientX - pan.x);
  pan.scroll.scrollTop = pan.top - (event.clientY - pan.y);
});
const endPan = (event) => {
  if (!pan || event.pointerId !== pan.id) return;
  pan.scroll.classList.remove("panning");
  pan = null;
};
app.addEventListener("pointerup", endPan);
app.addEventListener("pointercancel", endPan);
// Ctrl/⌘ + wheel (or a trackpad pinch) zooms the diagram around the pointer.
app.addEventListener(
  "wheel",
  (event) => {
    const scroll = event.target.closest(".graph-scroll");
    if (!scroll || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    const rect = scroll.getBoundingClientRect();
    zoomGraph(Math.exp(-event.deltaY * 0.0015), { x: event.clientX - rect.left, y: event.clientY - rect.top });
  },
  { passive: false },
);

const flushSession = () => {
  if (state.model) saveSession();
};
window.addEventListener("pagehide", flushSession);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flushSession();
});

window.addEventListener("resize", () => {
  applyZoom();
  fitChartText();
});
app.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-node-index]")) {
    event.preventDefault();
    event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  }
});
document.getElementById("file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 900000) throw new Error("Choose a model file smaller than 900 KB.");
    state.draft = await file.text();
    state.error = "";
    state.editorMessage = "Imported. Review the model, then Validate & apply to use it.";
    try {
      loadIntoBuilder(JSON.parse(state.draft));
      state.builderDirty = true;
      state.editorView = "builder";
    } catch (error) {
      state.builder = null;
      state.builderDirty = false;
      state.editorView = "json";
      state.error = `The imported file cannot be opened in the visual builder: ${error.message}`;
    }
    refreshBuilder();
  } catch (error) {
    toast(error.message);
  }
  event.target.value = "";
});

(async () => {
  try {
    state.config = await api("/api/config").catch(() => ({ public: false }));
    state.examples = await api("/api/examples");
    const session = loadSession();
    if (session) {
      let restored = false;
      try {
        await restoreSession(session);
        restored = true;
      } catch {
        clearSession(); // unreadable: start fresh rather than fail on every visit
      }
      if (restored) {
        toast("Restored your last session. Choose an example to start fresh.");
        return;
      }
    }
    await loadExample("umbrella");
  } catch (error) {
    app.innerHTML =
      `<div class="loading-page"><p>Unable to open BayesLab: ${escapeHtml(error.message)}` +
      " Start the Python server and reload this page.</p></div>";
  }
})();
