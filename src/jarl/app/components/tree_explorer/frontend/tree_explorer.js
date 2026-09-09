const LAYOUT_STORAGE_KEY = "jarl_tree_layout_mode";

function normalizeLayoutMode(value) {
  return value === "horizontal" ? "horizontal" : "vertical";
}

function loadLayoutMode() {
  try {
    return normalizeLayoutMode(window.localStorage.getItem(LAYOUT_STORAGE_KEY));
  } catch (error) {
    return "vertical";
  }
}

function saveLayoutMode(mode) {
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, normalizeLayoutMode(mode));
  } catch (error) {
    // Storage can be unavailable in strict/private browser contexts.
  }
}

const state = {
  payload: null,
  expanded: new Set(),
  expandedInitialized: false,
  knownNodeIds: new Set(),
  rootId: null,
  lastAutoFocusedNodeId: null,
  zoom: 1,
  search: "",
  selectedNodeId: null,
  contextNodeId: null,
  layoutMode: loadLayoutMode(),
};

const MIN_VIEWPORT_HEIGHT = 420;
const MAX_VIEWPORT_HEIGHT = 800;
const FRAME_VERTICAL_PADDING = 2;

const toolbar = document.getElementById("toolbar");
const canvas = document.getElementById("canvas");
const viewport = document.getElementById("viewport");
const tooltip = document.getElementById("tooltip");
const selectionHint = document.getElementById("selection-hint");
const contextMenu = document.getElementById("context-menu");
const searchInput = document.getElementById("search");
const zoomLabel = document.getElementById("zoom-label");
const layoutButton = document.getElementById("layout-toggle");
const focusButton = document.getElementById("focus-current");

const StreamlitBridge = {
  isActive() {
    return window.parent !== window;
  },
  init(onRender) {
    if (!this.isActive()) {
      return false;
    }
    window.addEventListener("message", (event) => {
      const data = event.data;
      if (!data || data.type !== "streamlit:render") {
        return;
      }
      const args = data.args || {};
      onRender(args.payload || {});
    });
    this.post({ type: "streamlit:componentReady", apiVersion: 1 });
    return true;
  },
  post(message) {
    window.parent.postMessage({ isStreamlitMessage: true, ...message }, "*");
  },
  setComponentValue(value) {
    this.post({ type: "streamlit:setComponentValue", value });
  },
  setFrameHeight(height) {
    this.post({ type: "streamlit:setFrameHeight", height });
  },
};

function metricColor(value, nodes) {
  const values = nodes
    .map((node) => node.metric_value)
    .filter((item) => item !== null && item !== undefined);
  if (value === null || value === undefined || values.length === 0) {
    return "var(--panel)";
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return "rgba(91, 141, 239, 0.25)";
  }
  const t = (value - min) / (max - min);
  const hue = Math.round(210 - t * 150);
  return `hsla(${hue}, 70%, 45%, 0.28)`;
}

const BRANCH_COLORS = [
  "#74d88f",
  "#58c7f3",
  "#f2a65a",
  "#d66ba0",
  "#b98cff",
  "#5eead4",
  "#ff8a7a",
  "#c7d95b",
  "#7cc7ff",
  "#f4c95d",
  "#9bd48b",
  "#f48fb1",
  "#80cbc4",
  "#ffb74d",
  "#90caf9",
  "#ce93d8",
  "#a5d6a7",
  "#ffab91",
  "#b0bec5",
  "#e6ee9c",
  "#81d4fa",
  "#bcaaa4",
  "#ffd54f",
  "#b39ddb",
];

function buildBranchColors(nodes) {
  const orderedBranches = [];
  const seen = new Set();
  for (const node of nodes) {
    if (!seen.has(node.branch)) {
      seen.add(node.branch);
      orderedBranches.push(node.branch);
    }
  }

  const branchColors = new Map();
  let nextColorIndex = seen.has("main") ? 1 : 0;
  for (const branch of orderedBranches) {
    if (branch === "main") {
      branchColors.set(branch, BRANCH_COLORS[0]);
      continue;
    }
    branchColors.set(branch, BRANCH_COLORS[nextColorIndex % BRANCH_COLORS.length]);
    nextColorIndex += 1;
  }
  return branchColors;
}

function branchColor(branch, branchColors) {
  return branchColors.get(branch) || BRANCH_COLORS[0];
}

function colorWithAlpha(hex, alpha) {
  const normalized = hex.replace("#", "");
  const red = parseInt(normalized.slice(0, 2), 16);
  const green = parseInt(normalized.slice(2, 4), 16);
  const blue = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function formatMetric(name, value) {
  if (value === null || value === undefined) {
    return `${name}: —`;
  }
  return `${name}: ${Number(value).toFixed(4)}`;
}

function childrenMap(payload) {
  const map = new Map();
  for (const [parentId, childId] of payload.edges || []) {
    if (!map.has(parentId)) {
      map.set(parentId, []);
    }
    map.get(parentId).push(childId);
  }
  for (const [parentId, childIds] of map.entries()) {
    childIds.sort();
  }
  return map;
}

function nodeById(payload) {
  const map = new Map();
  for (const node of payload.nodes || []) {
    map.set(node.id, node);
  }
  return map;
}

function addBranchHeadValue(headIds, value) {
  if (typeof value === "string") {
    headIds.add(value);
    return;
  }
  if (value && typeof value === "object" && typeof value.id === "string") {
    headIds.add(value.id);
  }
}

function branchHeadIds(payload, lookup) {
  const headIds = new Set();
  const branchHeads = payload.branch_heads || payload.branchHeads;
  if (Array.isArray(branchHeads)) {
    for (const value of branchHeads) {
      addBranchHeadValue(headIds, value);
    }
  } else if (branchHeads && typeof branchHeads === "object") {
    for (const value of Object.values(branchHeads)) {
      addBranchHeadValue(headIds, value);
    }
  }
  if (headIds.size > 0) {
    return headIds;
  }

  const sameBranchParents = new Set();
  for (const [parentId, childId] of payload.edges || []) {
    const parent = lookup.get(parentId);
    const child = lookup.get(childId);
    if (parent && child && parent.branch === child.branch) {
      sameBranchParents.add(parentId);
    }
  }
  for (const node of payload.nodes || []) {
    if (!sameBranchParents.has(node.id)) {
      headIds.add(node.id);
    }
  }
  return headIds;
}

function edgeKind(node, lookup) {
  if (!node.parent_id) {
    return "root";
  }
  const parent = lookup.get(node.parent_id);
  if (!parent) {
    return "extend";
  }
  return parent.branch === node.branch ? "extend" : "fork";
}

function hasForkChild(node, childIds, lookup) {
  for (const childId of childIds) {
    const child = lookup.get(childId);
    if (child && child.branch !== node.branch) {
      return true;
    }
  }
  return false;
}

function syncExpandedFromPayload(payload) {
  const nodeIds = new Set((payload.nodes || []).map((node) => node.id));
  const nextExpanded = new Set();

  if (!state.expandedInitialized) {
    for (const nodeId of nodeIds) {
      nextExpanded.add(nodeId);
    }
  } else {
    for (const nodeId of state.expanded) {
      if (nodeIds.has(nodeId)) {
        nextExpanded.add(nodeId);
      }
    }
    for (const nodeId of nodeIds) {
      if (!state.knownNodeIds.has(nodeId)) {
        nextExpanded.add(nodeId);
      }
    }
  }

  if (payload.root_id && (!state.expandedInitialized || state.rootId !== payload.root_id)) {
    nextExpanded.add(payload.root_id);
  }
  state.expanded = nextExpanded;
  state.knownNodeIds = nodeIds;
  state.rootId = payload.root_id || null;
  state.expandedInitialized = true;
}

function updateSelectionHint(message) {
  if (!selectionHint) {
    return;
  }
  if (!state.selectedNodeId) {
    selectionHint.hidden = true;
    return;
  }
  selectionHint.hidden = false;
  selectionHint.textContent = message || `Checkout: ${state.selectedNodeId}`;
}

function sendEvent(action, nodeId) {
  state.selectedNodeId = nodeId;
  updateSelectionHint();
  if (StreamlitBridge.isActive()) {
    StreamlitBridge.setComponentValue({
      action,
      node_id: nodeId,
    });
    return;
  }
  updateSelectionHint(`Seleccionado: ${nodeId}`);
}

function hideContextMenu() {
  if (!contextMenu) {
    return;
  }
  contextMenu.hidden = true;
  state.contextNodeId = null;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function resizeViewportToContent() {
  const toolbarHeight = toolbar ? toolbar.offsetHeight : 48;
  const contentHeight = Math.ceil(canvas.scrollHeight * state.zoom);
  const viewportHeight = clamp(contentHeight + 12, MIN_VIEWPORT_HEIGHT, MAX_VIEWPORT_HEIGHT);
  viewport.style.height = `${viewportHeight}px`;
  StreamlitBridge.setFrameHeight(toolbarHeight + viewportHeight + FRAME_VERTICAL_PADDING);
}

function updateLayoutButton() {
  if (!layoutButton) {
    return;
  }
  const isHorizontal = state.layoutMode === "horizontal";
  const title = isHorizontal ? "Cambiar a vista vertical" : "Cambiar a vista horizontal";
  layoutButton.textContent = isHorizontal ? "↕" : "↔";
  layoutButton.title = title;
  layoutButton.setAttribute("aria-label", title);
  layoutButton.setAttribute("aria-pressed", String(isHorizontal));
}

function applyLayoutMode(shouldRefocus = false) {
  document.body.dataset.layout = state.layoutMode;
  updateLayoutButton();
  resizeViewportToContent();
  if (shouldRefocus) {
    requestAnimationFrame(() => focusActiveNode("smooth"));
  }
}

function toggleLayoutMode() {
  state.layoutMode = state.layoutMode === "horizontal" ? "vertical" : "horizontal";
  saveLayoutMode(state.layoutMode);
  applyLayoutMode(true);
}

function nodeElementById(nodeId) {
  for (const element of canvas.querySelectorAll("[data-node-id]")) {
    if (element.getAttribute("data-node-id") === nodeId) {
      return element;
    }
  }
  return null;
}

function focusNode(nodeId, behavior = "smooth") {
  const element = nodeElementById(nodeId);
  if (!element) {
    return;
  }
  const viewportRect = viewport.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();
  const targetLeft = viewport.scrollLeft + elementRect.left - viewportRect.left - viewport.clientWidth * 0.24;
  const targetTop = viewport.scrollTop + elementRect.top - viewportRect.top - viewport.clientHeight * 0.28;
  viewport.scrollTo({
    left: Math.max(0, targetLeft),
    top: Math.max(0, targetTop),
    behavior,
  });
}

function focusActiveNode(behavior = "smooth") {
  const nodeId = state.selectedNodeId || (state.payload && state.payload.current_node_id);
  if (nodeId) {
    focusNode(nodeId, behavior);
  }
}

function autoFocusCurrentNode(nodeId) {
  if (!nodeId || state.lastAutoFocusedNodeId === nodeId) {
    return;
  }
  state.lastAutoFocusedNodeId = nodeId;
  requestAnimationFrame(() => focusNode(nodeId, "smooth"));
}

function renderLegend() {
  return `
    <div class="graph-legend">
      <span class="legend-chip legend-extend">EXTEND</span>
      <span class="legend-chip legend-fork">FORK</span>
    </div>`;
}

function renderTree(payload) {
  if (!payload || !payload.nodes || payload.nodes.length === 0) {
    canvas.innerHTML = '<div class="empty-state">Sin nodos visibles.</div>';
    StreamlitBridge.setFrameHeight(180);
    return;
  }

  const nodes = payload.nodes;
  const lookup = nodeById(payload);
  const children = childrenMap(payload);
  const heads = branchHeadIds(payload, lookup);
  const branchColors = buildBranchColors(nodes);
  const search = state.search.trim().toLowerCase();

  function renderNode(nodeId) {
    const node = lookup.get(nodeId);
    if (!node) {
      return "";
    }
    const childIds = children.get(nodeId) || [];
    const hasChildren = childIds.length > 0;
    const isExpanded = state.expanded.has(nodeId);
    const isCurrent = nodeId === payload.current_node_id;
    const isSelected = nodeId === state.selectedNodeId;
    const isMatch = search && nodeId.toLowerCase().includes(search);
    const isRoot = nodeId === payload.root_id;
    const nodeEdgeKind = edgeKind(node, lookup);
    const parent = node.parent_id ? lookup.get(node.parent_id) : null;
    const isForkSource = hasForkChild(node, childIds, lookup);
    const isBranchHead = heads.has(nodeId);
    const branchColorValue = branchColor(node.branch, branchColors);
    const parentBranchColorValue = parent ? branchColor(parent.branch, branchColors) : branchColorValue;
    const classes = ["node-card"];
    if (isForkSource) classes.push("fork-source");
    if (isBranchHead) classes.push("branch-head");
    if (isRoot) classes.push("root-node");
    if (isCurrent) classes.push("current");
    if (isSelected) classes.push("selected");
    if (isMatch) classes.push("search-match");
    const itemClasses = ["tree-item", `edge-${nodeEdgeKind}`];

    const metricBadge = payload.metric_key
      ? `<span class="badge">${formatMetric(payload.metric_key, node.metric_value)}</span>`
      : "";
    const headBadge = isBranchHead ? '<span class="badge head-badge">HEAD</span>' : "";
    const rootBadge = isRoot ? '<span class="badge root-badge">ROOT</span>' : "";
    const opBadge =
      nodeEdgeKind === "root" ? "" : `<span class="badge op-badge ${nodeEdgeKind}-badge">${nodeEdgeKind}</span>`;
    const forkSourceBadge = isForkSource ? '<span class="badge fork-source-badge">FORKS</span>' : "";
    const toggle = hasChildren
      ? `<button class="toggle" data-toggle="${nodeId}" type="button" aria-label="${isExpanded ? "Contraer" : "Expandir"} ${nodeId}" aria-expanded="${isExpanded}">${isExpanded ? "−" : "+"}</button>`
      : '<span class="toggle placeholder">·</span>';
    const inlineStyle = [
      `--branch-color: ${branchColorValue}`,
      `--branch-soft: ${colorWithAlpha(branchColorValue, 0.18)}`,
      `--branch-faint: ${colorWithAlpha(branchColorValue, 0.08)}`,
      `--parent-branch-color: ${parentBranchColorValue}`,
      `--parent-branch-soft: ${colorWithAlpha(parentBranchColorValue, 0.2)}`,
      `--parent-branch-faint: ${colorWithAlpha(parentBranchColorValue, 0.08)}`,
      `--edge-color: ${parentBranchColorValue}`,
      `--metric-color: ${metricColor(node.metric_value, nodes)}`,
    ].join("; ");

    const childrenHtml =
      hasChildren && isExpanded
        ? `<ul class="tree-children">${childIds.map((childId) => renderNode(childId)).join("")}</ul>`
        : "";

    return `
      <li class="${itemClasses.join(" ")}" style="${inlineStyle}">
        <div class="node-row">
          <span class="graph-dot" aria-hidden="true"></span>
          ${toggle}
          <div
            class="${classes.join(" ")}"
            data-node-id="${nodeId}"
          >
            <div class="node-title">
              <div class="node-id">${node.id}</div>
              <span class="branch-chip">${node.branch}</span>
            </div>
            <div class="node-meta">${node.branch} · ${node.label || "—"} · ${node.status} · step ${node.step}</div>
            <div class="node-metrics">
              <span class="badge status-${node.status}">${node.status}</span>
              ${rootBadge}
              ${opBadge}
              ${forkSourceBadge}
              ${headBadge}
              ${metricBadge}
              <span class="badge">${formatMetric("train", node.train_return)}</span>
              <span class="badge">${formatMetric("eval", node.eval_return)}</span>
            </div>
          </div>
        </div>
        ${childrenHtml}
      </li>`;
  }

  canvas.innerHTML = `${renderLegend()}<ul class="tree-root">${renderNode(payload.root_id)}</ul>`;
  applyZoom();
  autoFocusCurrentNode(payload.current_node_id);
}

function applyZoom() {
  canvas.style.transform = `scale(${state.zoom})`;
  zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  resizeViewportToContent();
}

function showTooltip(node, lookup, x, y) {
  const highlights = node.highlights || {};
  const parent = node.parent_id ? lookup.get(node.parent_id) : null;
  const operation = parent ? (parent.branch === node.branch ? "extend" : "fork") : "root";
  const lines = [
    `<strong>${node.id}</strong>`,
    `${node.branch} · ${node.label || "—"}`,
    `op: ${operation}${parent ? ` · parent: ${parent.id}` : ""}`,
    `status: ${node.status} · step: ${node.step}`,
    formatMetric("train", node.train_return),
    formatMetric("eval", node.eval_return),
  ];
  for (const [key, value] of Object.entries(highlights)) {
    lines.push(`${key}: ${value}`);
  }
  tooltip.innerHTML = lines.join("<br/>");
  tooltip.hidden = false;
  tooltip.style.left = `${x + 14}px`;
  tooltip.style.top = `${y + 14}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function boot(payload) {
  state.payload = payload || {};
  syncExpandedFromPayload(state.payload);
  if (state.payload.search_filter) {
    state.search = state.payload.search_filter;
    searchInput.value = state.search;
  }
  state.selectedNodeId = state.payload.current_node_id || state.selectedNodeId;
  updateSelectionHint(state.selectedNodeId ? `Nodo activo: ${state.selectedNodeId}` : "");
  renderTree(state.payload);
}

searchInput.addEventListener("input", () => {
  state.search = searchInput.value;
  renderTree(state.payload);
});

document.getElementById("zoom-in").addEventListener("click", () => {
  state.zoom = Math.min(state.zoom + 0.1, 2.5);
  applyZoom();
});

document.getElementById("zoom-out").addEventListener("click", () => {
  state.zoom = Math.max(state.zoom - 0.1, 0.4);
  applyZoom();
});

document.getElementById("zoom-reset").addEventListener("click", () => {
  state.zoom = 1;
  applyZoom();
});

if (focusButton) {
  focusButton.addEventListener("click", () => focusActiveNode());
}

if (layoutButton) {
  layoutButton.addEventListener("click", toggleLayoutMode);
}

window.addEventListener("storage", (event) => {
  if (event.key !== LAYOUT_STORAGE_KEY) {
    return;
  }
  state.layoutMode = normalizeLayoutMode(event.newValue);
  applyLayoutMode(true);
});

let dragActive = false;
let dragStartX = 0;
let dragStartY = 0;
let scrollStartX = 0;
let scrollStartY = 0;
let dragMoved = false;
let suppressNextClick = false;

function canStartViewportDrag(event) {
  if (event.button !== 0 && event.button !== 1) {
    return false;
  }
  if (event.target.closest(".toggle, #search, #context-menu, #zoom-controls")) {
    return false;
  }
  return event.button === 1 || event.shiftKey || !event.target.closest(".node-card");
}

viewport.addEventListener("mousedown", (event) => {
  if (!canStartViewportDrag(event)) {
    return;
  }
  event.preventDefault();
  dragActive = true;
  dragMoved = false;
  viewport.classList.add("dragging");
  dragStartX = event.clientX;
  dragStartY = event.clientY;
  scrollStartX = viewport.scrollLeft;
  scrollStartY = viewport.scrollTop;
});

window.addEventListener("mousemove", (event) => {
  if (!dragActive) {
    return;
  }
  if (Math.abs(event.clientX - dragStartX) > 4 || Math.abs(event.clientY - dragStartY) > 4) {
    dragMoved = true;
  }
  viewport.scrollLeft = scrollStartX - (event.clientX - dragStartX);
  viewport.scrollTop = scrollStartY - (event.clientY - dragStartY);
});

window.addEventListener("mouseup", () => {
  if (dragActive && dragMoved) {
    suppressNextClick = true;
    window.setTimeout(() => {
      suppressNextClick = false;
    }, 0);
  }
  dragActive = false;
  viewport.classList.remove("dragging");
});

viewport.addEventListener(
  "wheel",
  (event) => {
    if (!event.shiftKey || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return;
    }
    viewport.scrollLeft += event.deltaY;
    event.preventDefault();
  },
  { passive: false }
);

canvas.addEventListener("click", (event) => {
  if (suppressNextClick) {
    event.preventDefault();
    suppressNextClick = false;
    return;
  }
  const toggle = event.target.closest("[data-toggle]");
  if (toggle) {
    const nodeId = toggle.getAttribute("data-toggle");
    if (state.expanded.has(nodeId)) {
      state.expanded.delete(nodeId);
    } else {
      state.expanded.add(nodeId);
    }
    renderTree(state.payload);
    return;
  }
  const card = event.target.closest("[data-node-id]");
  if (card) {
    hideContextMenu();
    sendEvent("checkout", card.getAttribute("data-node-id"));
    renderTree(state.payload);
  }
});

canvas.addEventListener("contextmenu", (event) => {
  const card = event.target.closest("[data-node-id]");
  if (!card || !contextMenu) {
    return;
  }
  event.preventDefault();
  state.contextNodeId = card.getAttribute("data-node-id");
  contextMenu.hidden = false;
  contextMenu.style.left = `${event.clientX}px`;
  contextMenu.style.top = `${event.clientY}px`;
});

canvas.addEventListener("mouseover", (event) => {
  const card = event.target.closest("[data-node-id]");
  if (!card || !state.payload) {
    hideTooltip();
    return;
  }
  const nodeId = card.getAttribute("data-node-id");
  const lookup = nodeById(state.payload);
  const node = lookup.get(nodeId);
  if (node) {
    showTooltip(node, lookup, event.clientX, event.clientY);
  }
});

canvas.addEventListener("mouseout", (event) => {
  if (!event.relatedTarget || !event.relatedTarget.closest("[data-node-id]")) {
    hideTooltip();
  }
});

if (contextMenu) {
  contextMenu.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button || !state.contextNodeId) {
      return;
    }
    sendEvent(button.getAttribute("data-action"), state.contextNodeId);
    hideContextMenu();
  });
}

document.addEventListener("click", (event) => {
  if (!event.target.closest("#context-menu")) {
    hideContextMenu();
  }
});

applyLayoutMode();

if (!StreamlitBridge.init(boot) && window.JARL_TREE_PAYLOAD) {
  boot(window.JARL_TREE_PAYLOAD);
}
