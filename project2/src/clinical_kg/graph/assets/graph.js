// Cytoscape is supplied by graph_view.py from the pinned local UMD bundle.
// Keep a small viewport cache across Streamlit reruns; dispose every canvas.
const viewports = new Map();

export default function renderGraph({ data, parentElement, setStateValue }) {
  const canvas = parentElement.querySelector(".kg-canvas");
  const picker = parentElement.querySelector(".kg-picker");
  const search = parentElement.querySelector(".kg-search");
  const root = parentElement.querySelector(".kg-root");
  const tooltip = parentElement.querySelector(".kg-tooltip");
  tooltip.hidden = true;
  const status = parentElement.querySelector(".kg-status");
  const previous = viewports.get(data.fingerprint);
  const initialIds = new Set(data.views[previous?.selectedId] || data.views[""]);
  const categories = data.elements.filter(e => e.data.kind === "SemanticType");
  const memberships = new Map();
  data.elements.filter(e => e.data.relation === "IS_A").forEach(e => {
    if (!memberships.has(e.data.source)) memberships.set(e.data.source, e.data.target);
  });
  const groupedPositions = {};
  data.elements.filter(e => ["Person", "Age", "Sex", "Article"].includes(e.data.kind))
    .forEach((e, i) => { groupedPositions[e.data.id] = { x: i * 180, y: -400 }; });
  categories.forEach((category, index) => {
    const center = { x: (index % 3) * 1000, y: Math.floor(index / 3) * 1000 };
    groupedPositions[category.data.id] = center;
    const members = [...memberships].filter(([, target]) => target === category.data.id);
    members.forEach(([id], i) => {
      const ring = Math.floor(i / 12);
      const angle = 2 * Math.PI * (i % 12) / Math.min(12, members.length - ring * 12) + ring * 0.25;
      const radius = 190 + 110 * ring;
      groupedPositions[id] = { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) };
    });
  });
  const columns = Math.ceil(Math.sqrt(data.elements.filter(e => !e.data.source).length));
  const elements = data.elements.map((element, index) => ({
    ...element,
    // Seed the force layout deterministically; coincident (0,0) nodes produce
    // poor layouts when the component first mounts in a hidden tab.
    position: previous?.positions[element.data.id] || (data.grouped && groupedPositions[element.data.id]) || {
      x: (index % columns) * 160, y: Math.floor(index / columns) * 100,
    },
  }));
  const layout = {
    name: data.layout,
    animate: false,
    padding: 45,
    nodeDimensionsIncludeLabels: true,
    boundingBox: { x1: 0, y1: 0, w: canvas.clientWidth || 900, h: 580 },
    ...(data.layout === "breadthfirst" ? { directed: true, spacingFactor: 1.2 } : {}),
    ...(data.layout === "cose" ? { randomize: false, nodeRepulsion: 9000, idealEdgeLength: 130, nodeOverlap: 30 } : {}),
  };
  const cy = cytoscape({
    container: canvas, elements: elements.filter(e => initialIds.has(e.data.id)), style: data.style,
    layout: previous || data.grouped ? { name: "preset", fit: !previous, padding: 45 } : layout,
    minZoom: 0.12, maxZoom: 3, wheelSensitivity: 0.2,
    selectionType: "single", boxSelectionEnabled: false,
  });
  if (previous) {
    cy.zoom(previous.zoom);
    cy.pan(previous.pan);
  }

  // Case text and labels are data, never HTML or executable JavaScript.
  function updatePicker() {
    picker.replaceChildren(new Option("Select a node to inspect its evidence…", ""));
    const query = search.value.trim().toLocaleLowerCase();
    cy.nodes().filter(n => !query || `${n.data("label")} ${n.data("cui") || ""}`.toLocaleLowerCase().includes(query))
      .sort((a, b) => a.data("label").localeCompare(b.data("label"))).forEach(node => {
      picker.add(new Option(`${node.data("kind")} · ${node.data("label")}`, node.id()));
    });
  }
  updatePicker();
  const legend = parentElement.querySelector(".kg-legend");
  const types = new Map();
  data.elements.filter(e => ["Concept", "Mention"].includes(e.data.kind)).forEach(e => types.set(e.data.entity_type, e.data.color));
  legend.replaceChildren();
  [...types].sort().forEach(([type, color]) => {
    const chip = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.backgroundColor = color;
    chip.append(dot, document.createTextNode(type));
    legend.append(chip);
  });
  function updateDetail() {
    cy.nodes().toggleClass("kg-overview", cy.zoom() < 0.55);
    cy.nodes('[kind = "SemanticType"], [kind = "Person"]')
      .style("font-size", Math.min(44, 12 / cy.zoom()));
  }
  cy.on("zoom", updateDetail);
  updateDetail();
  let selectedId = previous?.selectedId || "";
  let restoring = false;
  function select(id, emit = true) {
    tooltip.hidden = true;
    const ids = new Set(data.views[id] || data.views[""]);
    cy.batch(() => {
      cy.elements().filter(e => !ids.has(e.id())).remove();
      const additions = elements.filter(e => ids.has(e.data.id) && !cy.getElementById(e.data.id).length);
      cy.add(additions);
      additions.filter(e => e.data.kind === "Measurement").forEach((e, index) => {
        const node = cy.getElementById(e.data.id);
        const source = node.incomers("node")[0];
        if (source) {
          const position = source.position();
          const angle = index * 2.4;
          const radius = 140 + 55 * Math.floor(index / 6);
          node.position({ x: position.x + radius * Math.cos(angle), y: position.y + radius * Math.sin(angle) });
        }
      });
    });
    updatePicker();
    updateDetail();
    const element = cy.getElementById(id);
    restoring = true;
    cy.elements().unselect();
    cy.elements().removeClass("kg-muted");
    if (element.length) element.select();
    if (element.length && !["Person", "Article"].includes(element.data("kind"))) {
      const neighborhood = element.isNode() ? element.closedNeighborhood()
        : element.union(element.source()).union(element.target());
      cy.elements().difference(neighborhood).addClass("kg-muted");
    }
    restoring = false;
    selectedId = element.length ? id : "";
    picker.value = element.isNode() ? selectedId : "";
    status.textContent = element.length ? element.data("tooltip") : "Click a node or edge to inspect evidence.";
    if (emit) setStateValue("selection", { id: selectedId, fingerprint: data.fingerprint });
  }
  if (selectedId) select(selectedId, false);
  picker.onchange = () => select(picker.value);
  search.oninput = updatePicker;
  search.onkeydown = event => {
    if (event.key === "Enter" && picker.options.length > 1) {
      select(picker.options[1].value);
      focusSelection();
    }
  };
  root.onkeydown = event => {
    if (event.key === "Escape") { search.value = ""; select(""); }
  };
  cy.on("tap", "node, edge", event => {
    if (!restoring) select(event.target.id());
  });
  cy.on("tap", event => { if (event.target === cy) select(""); });
  cy.on("mouseover", "node, edge", event => {
    event.target.addClass("kg-hover");
    canvas.style.cursor = "pointer";
    tooltip.textContent = event.target.data("tooltip");
    tooltip.hidden = false;
  });
  cy.on("mouseout", "node, edge", event => {
    event.target.removeClass("kg-hover");
    canvas.style.cursor = "grab";
    tooltip.hidden = true;
  });

  const fit = parentElement.querySelector(".kg-fit");
  const reset = parentElement.querySelector(".kg-layout");
  const clear = parentElement.querySelector(".kg-clear");
  const focus = parentElement.querySelector(".kg-focus");
  const zoomIn = parentElement.querySelector(".kg-zoom-in");
  const zoomOut = parentElement.querySelector(".kg-zoom-out");
  function focusSelection() {
    const selected = cy.getElementById(selectedId);
    if (!selected.length) return;
    const neighborhood = selected.isNode() ? selected.closedNeighborhood() : selected.union(selected.connectedNodes());
    cy.fit(neighborhood, 90);
  }
  clear.onclick = () => { search.value = ""; select(""); };
  focus.onclick = focusSelection;
  function zoom(factor) {
    cy.zoom({ level: Math.max(cy.minZoom(), Math.min(cy.maxZoom(), cy.zoom() * factor)),
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  }
  zoomIn.onclick = () => zoom(1.3);
  zoomOut.onclick = () => zoom(1 / 1.3);
  fit.onclick = () => cy.fit(undefined, 45);
  reset.onclick = () => {
    if (data.grouped) {
      cy.nodes().forEach(n => { if (groupedPositions[n.id()]) n.position(groupedPositions[n.id()]); });
      cy.fit(undefined, 45);
    } else cy.layout(layout).run();
  };
  let fitted = Boolean(previous);
  const observer = new ResizeObserver(() => {
    const { width, height } = canvas.getBoundingClientRect();
    if (width && height) {
      cy.resize();
      if (!fitted) { cy.fit(undefined, 45); fitted = true; }
    }
  });
  observer.observe(canvas);

  return () => {
    viewports.set(data.fingerprint, {
      positions: Object.fromEntries(cy.nodes().map(n => [n.id(), { ...n.position() }])),
      zoom: cy.zoom(), pan: { ...cy.pan() }, selectedId,
    });
    if (viewports.size > 8) viewports.delete(viewports.keys().next().value);
    observer.disconnect();
    picker.onchange = null;
    search.oninput = search.onkeydown = root.onkeydown = null;
    clear.onclick = focus.onclick = zoomIn.onclick = zoomOut.onclick = null;
    fit.onclick = reset.onclick = null;
    cy.destroy();
  };
}
