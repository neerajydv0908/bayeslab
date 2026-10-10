/* BayesLab visual model builder: model edits as plain functions, plus the form view.
   Loaded before app.js. Every edit keeps the model in the same JSON format the
   Python engine validates, so the builder and the JSON editor stay in sync. */
"use strict";

const Builder = (() => {
  const MAX_NODES = 30;
  const MIN_STATES = 2;
  const MAX_STATES = 8;
  const MAX_NAME = 80;
  const MAX_ROWS_SHOWN = 400;

  const esc = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
    );
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  // Own keys only: a node or state may be called "constructor" or "toString".
  const owns = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  // Rows that were missing from an imported model and were filled with defaults.
  // They stay flagged until the user edits or normalizes them.
  const filledRows = new WeakSet();
  const product = (lists) =>
    lists.reduce((acc, list) => acc.flatMap((prefix) => list.map((item) => [...prefix, item])), [[]]);

  function domains(model) {
    const map = new Map();
    for (const node of model.chance_nodes) map.set(node.name, node.states);
    for (const node of model.decision_nodes) map.set(node.name, node.states);
    return map;
  }

  function allNames(model) {
    return [...model.chance_nodes, ...model.decision_nodes, ...model.utility_nodes].map((node) => node.name);
  }

  function find(model, name) {
    for (const kind of ["chance", "decision", "utility"]) {
      const node = model[`${kind}_nodes`].find((item) => item.name === name);
      if (node) return { node, kind };
    }
    return { node: null, kind: null };
  }

  // A row matches an assignment when every variable it names has the assigned
  // state. Variables a row names but the assignment lacks match `fallback`
  // (used when a parent is removed) or anything at all.
  function findRow(rows, assignment, fallback = {}) {
    return rows.find(
      (row) =>
        isObject(row) &&
        isObject(row.when) &&
        Object.entries(row.when).every(([name, state]) =>
          owns(assignment, name) ? assignment[name] === state : !owns(fallback, name) || fallback[name] === state,
        ),
    );
  }

  // Rebuild a table with one row per parent assignment, in canonical order,
  // copying values from the closest existing row. `lookup` may redirect an
  // assignment (for a new state, copy the rows of the first state).
  function rebuild(model, parents, rows, makeRow, lookup = (a) => a, fallback = {}, onMissing = null) {
    const dom = domains(model);
    const size = parents.reduce((total, name) => total * (dom.get(name)?.length || 0), 1);
    if (size > 25000) throw new Error("That table would need more than 25,000 rows. Use fewer parents or states.");
    return product(parents.map((name) => dom.get(name) || [])).map((combo) => {
      const when = Object.fromEntries(parents.map((name, i) => [name, combo[i]]));
      const old = findRow(rows, lookup(when), fallback);
      const row = makeRow(when, old);
      if (!old && onMissing) onMissing(row);
      else if (old && filledRows.has(old)) filledRows.add(row); // still unreviewed after an edit elsewhere
      return row;
    });
  }

  const cptRow = (node) => (when, old) => {
    const probabilities = {};
    for (const state of node.states) {
      if (old && isObject(old.probabilities)) probabilities[state] = owns(old.probabilities, state) ? old.probabilities[state] : 0;
      else probabilities[state] = 1 / node.states.length;
    }
    return { when, probabilities };
  };
  const utilityRow = (when, old) => ({ when, value: old && owns(old, "value") ? old.value : 0 });

  function rebuildNode(model, node, kind, lookup, fallback, onMissing) {
    if (kind === "chance")
      node.cpt = rebuild(model, node.parents, node.cpt, cptRow(node), lookup, fallback, onMissing);
    if (kind === "utility")
      node.table = rebuild(model, node.parents, node.table, utilityRow, lookup, fallback, onMissing);
  }

  function children(model, name) {
    return [
      ...model.chance_nodes.filter((node) => node.parents.includes(name)).map((node) => ({ node, kind: "chance" })),
      ...model.utility_nodes.filter((node) => node.parents.includes(name)).map((node) => ({ node, kind: "utility" })),
    ];
  }

  /** Copy a model into the builder's complete, canonical form, or explain why it cannot be edited here.

      Missing table rows are filled (uniform probabilities, utility 0), flagged for review, and
      listed in `report` ({node: count}) so the caller can say so. */
  function normalize(raw, report = null) {
    if (!isObject(raw)) throw new Error("The model must be a JSON object.");
    for (const key of ["chance_nodes", "decision_nodes", "utility_nodes"])
      if (raw[key] !== undefined && !Array.isArray(raw[key])) throw new Error(`${key} must be a list.`);
    if ((raw.decision_nodes || []).length > 1) throw new Error("The builder supports one decision node.");
    const seen = new Set();
    const nameOf = (node, kind) => {
      if (!isObject(node) || typeof node.name !== "string" || !node.name.trim())
        throw new Error(`Every ${kind} node needs a name.`);
      if (seen.has(node.name)) throw new Error(`Two nodes are named ${node.name}.`);
      seen.add(node.name);
      return node.name;
    };
    const statesOf = (node) => {
      if (!Array.isArray(node.states) || node.states.some((state) => typeof state !== "string"))
        throw new Error(`${node.name} needs a list of state names.`);
      return [...node.states];
    };
    const listOf = (value) => (Array.isArray(value) ? value.filter((item) => typeof item === "string") : []);
    const model = {
      name: typeof raw.name === "string" && raw.name.trim() ? raw.name : "Untitled network",
      ...(typeof raw.description === "string" && raw.description ? { description: raw.description } : {}),
      chance_nodes: (raw.chance_nodes || []).map((node) => ({
        name: nameOf(node, "chance"),
        states: statesOf(node),
        parents: listOf(node.parents),
        cpt: Array.isArray(node.cpt) ? node.cpt : [],
      })),
      decision_nodes: (raw.decision_nodes || []).map((node) => ({
        name: nameOf(node, "decision"),
        states: statesOf(node),
        information: listOf(node.information),
        observation_costs: isObject(node.observation_costs) ? { ...node.observation_costs } : {},
      })),
      utility_nodes: (raw.utility_nodes || []).map((node) => ({
        name: nameOf(node, "utility"),
        parents: listOf(node.parents),
        table: Array.isArray(node.table) ? node.table : [],
      })),
    };
    const dom = domains(model);
    for (const node of [...model.chance_nodes, ...model.utility_nodes])
      for (const parent of node.parents)
        if (!dom.has(parent))
          throw new Error(`${node.name} lists ${parent} as a parent, but no chance or decision node has that name.`);
    const chanceNames = model.chance_nodes.map((node) => node.name);
    for (const decision of model.decision_nodes) {
      for (const name of decision.information)
        if (!chanceNames.includes(name))
          throw new Error(`${decision.name} observes ${name}, but there is no chance node with that name.`);
      for (const name of Object.keys(decision.observation_costs))
        if (!decision.information.includes(name)) delete decision.observation_costs[name];
    }
    const missing = (node) => (row) => {
      filledRows.add(row);
      if (report) report[node.name] = (report[node.name] || 0) + 1;
    };
    for (const node of model.chance_nodes) rebuildNode(model, node, "chance", undefined, undefined, missing(node));
    for (const node of model.utility_nodes) rebuildNode(model, node, "utility", undefined, undefined, missing(node));
    return model;
  }

  function checkName(model, value, current = null) {
    const name = String(value).trim();
    if (!name) throw new Error("Names cannot be empty.");
    if (name.length > MAX_NAME) throw new Error(`Names can be at most ${MAX_NAME} characters.`);
    if (name !== current && allNames(model).includes(name)) throw new Error(`Another node is already named ${name}.`);
    return name;
  }

  function uniqueName(model, base) {
    const names = new Set(allNames(model));
    let index = 1;
    while (names.has(`${base} ${index}`)) index++;
    return `${base} ${index}`;
  }

  function checkRoom(model) {
    if (allNames(model).length >= MAX_NODES) throw new Error(`A network can have at most ${MAX_NODES} nodes.`);
  }

  function addChance(model) {
    checkRoom(model);
    const name = uniqueName(model, "Variable");
    model.chance_nodes.push({
      name,
      states: ["yes", "no"],
      parents: [],
      cpt: [{ when: {}, probabilities: { yes: 0.5, no: 0.5 } }],
    });
    return name;
  }

  function addUtility(model) {
    checkRoom(model);
    const name = uniqueName(model, "Utility");
    const decision = model.decision_nodes[0];
    const node = { name, parents: decision ? [decision.name] : [], table: [] };
    model.utility_nodes.push(node);
    rebuildNode(model, node, "utility");
    return name;
  }

  function addDecision(model) {
    if (model.decision_nodes.length) throw new Error("This version supports one decision node.");
    checkRoom(model);
    const name = uniqueName(model, "Decision");
    model.decision_nodes.push({ name, states: ["act", "wait"], information: [], observation_costs: {} });
    // A decision needs a utility to compare its actions.
    if (!model.utility_nodes.length && allNames(model).length < MAX_NODES) addUtility(model);
    return name;
  }

  function setParents(model, name, parents) {
    const { node, kind } = find(model, name);
    if (!node || kind === "decision") return;
    const order = [...model.chance_nodes, ...model.decision_nodes].map((item) => item.name);
    const chosen = order.filter((item) => parents.includes(item) && item !== name);
    const dom = domains(model);
    const fallback = Object.fromEntries(
      node.parents.filter((p) => !chosen.includes(p)).map((p) => [p, dom.get(p)?.[0]]),
    );
    node.parents = chosen;
    rebuildNode(model, node, kind, undefined, fallback);
  }

  function removeNode(model, name) {
    const { kind } = find(model, name);
    if (!kind) return;
    if (kind !== "utility") {
      // Disconnect children first, while the node's states are still known:
      // each child keeps the rows for the removed parent's first state.
      for (const child of children(model, name))
        setParents(model, child.node.name, child.node.parents.filter((parent) => parent !== name));
      for (const decision of model.decision_nodes) {
        decision.information = decision.information.filter((item) => item !== name);
        delete decision.observation_costs[name];
      }
    }
    model[`${kind}_nodes`] = model[`${kind}_nodes`].filter((node) => node.name !== name);
  }

  function renameKeys(rows, oldName, newName) {
    for (const row of rows)
      if (isObject(row.when) && owns(row.when, oldName))
        row.when = Object.fromEntries(
          Object.entries(row.when).map(([key, value]) => [key === oldName ? newName : key, value]),
        );
  }

  function renameNode(model, oldName, value) {
    const { node } = find(model, oldName);
    if (!node) return oldName;
    const name = checkName(model, value, oldName);
    if (name === oldName) return name;
    node.name = name;
    for (const item of [...model.chance_nodes, ...model.utility_nodes]) {
      item.parents = item.parents.map((parent) => (parent === oldName ? name : parent));
      renameKeys(item.cpt || item.table, oldName, name);
    }
    for (const decision of model.decision_nodes) {
      decision.information = decision.information.map((item) => (item === oldName ? name : item));
      if (owns(decision.observation_costs, oldName)) {
        decision.observation_costs[name] = decision.observation_costs[oldName];
        delete decision.observation_costs[oldName];
      }
    }
    return name;
  }

  function checkState(node, value, current = null) {
    const state = String(value).trim();
    if (!state) throw new Error("State names cannot be empty.");
    if (state.length > MAX_NAME) throw new Error(`State names can be at most ${MAX_NAME} characters.`);
    if (state !== current && node.states.includes(state)) throw new Error(`${node.name} already has a state named ${state}.`);
    return state;
  }

  function addState(model, name) {
    const { node, kind } = find(model, name);
    if (!node || kind === "utility") return;
    if (node.states.length >= MAX_STATES) throw new Error(`A variable can have at most ${MAX_STATES} states.`);
    let index = node.states.length + 1;
    while (node.states.includes(`state ${index}`)) index++;
    const state = `state ${index}`;
    const first = node.states[0];
    node.states.push(state);
    if (kind === "chance") for (const row of node.cpt) row.probabilities[state] = 0;
    // Children copy the rows of the first state, so their tables stay complete.
    const redirect = (when) => (when[name] === state ? { ...when, [name]: first } : when);
    for (const child of children(model, name)) rebuildNode(model, child.node, child.kind, redirect);
  }

  function rowSum(row, states) {
    return states.reduce((total, state) => total + Number(row.probabilities[state]), 0);
  }

  function normalizeRow(node, row) {
    const values = node.states.map((state) => Number(row.probabilities[state]));
    const valid = values.every((value) => Number.isFinite(value) && value >= 0);
    const total = valid ? values.reduce((a, b) => a + b, 0) : 0;
    node.states.forEach((state, i) => {
      row.probabilities[state] = total > 0 ? Number((values[i] / total).toPrecision(12)) : 1 / node.states.length;
    });
    // Rounding can leave the row a hair away from 1; put the difference on the largest entry.
    const sum = rowSum(row, node.states);
    if (sum !== 1) {
      const largest = node.states.reduce((a, b) => (row.probabilities[a] >= row.probabilities[b] ? a : b));
      row.probabilities[largest] = Number((row.probabilities[largest] + 1 - sum).toPrecision(12));
    }
  }

  function removeState(model, name, state) {
    const { node, kind } = find(model, name);
    if (!node || kind === "utility" || !node.states.includes(state)) return;
    if (node.states.length <= MIN_STATES) throw new Error(`A variable needs at least ${MIN_STATES} states.`);
    node.states = node.states.filter((item) => item !== state);
    if (kind === "chance")
      for (const row of node.cpt) {
        delete row.probabilities[state];
        normalizeRow(node, row);
      }
    for (const child of children(model, name)) rebuildNode(model, child.node, child.kind);
  }

  function renameState(model, name, oldState, value) {
    const { node, kind } = find(model, name);
    if (!node || kind === "utility") return;
    const state = checkState(node, value, oldState);
    if (state === oldState) return;
    node.states = node.states.map((item) => (item === oldState ? state : item));
    if (kind === "chance")
      for (const row of node.cpt)
        row.probabilities = Object.fromEntries(
          Object.entries(row.probabilities).map(([key, v]) => [key === oldState ? state : key, v]),
        );
    for (const child of children(model, name))
      for (const row of child.node.cpt || child.node.table)
        if (row.when[name] === oldState) row.when[name] = state;
  }

  /** Nodes reachable from `name` along probability arcs, information arcs, and decision arcs. */
  function descendants(model, name) {
    const next = new Map();
    const link = (from, to) => next.set(from, [...(next.get(from) || []), to]);
    for (const node of model.chance_nodes) for (const parent of node.parents) link(parent, node.name);
    for (const decision of model.decision_nodes) for (const item of decision.information) link(item, decision.name);
    const found = new Set();
    const pending = [...(next.get(name) || [])];
    while (pending.length) {
      const item = pending.pop();
      if (!found.has(item)) {
        found.add(item);
        pending.push(...(next.get(item) || []));
      }
    }
    return found;
  }

  /** Candidate parents (or observations) with whether each would create a cycle. */
  function parentOptions(model, name) {
    const { kind } = find(model, name);
    const chance = model.chance_nodes.map((node) => node.name);
    const decision = model.decision_nodes.map((node) => node.name);
    if (kind === "utility") return [...chance, ...decision].map((item) => ({ name: item, cycle: false }));
    const below = descendants(model, name);
    const pool = kind === "decision" ? chance : [...chance, ...decision];
    return pool.filter((item) => item !== name).map((item) => ({ name: item, cycle: below.has(item) }));
  }

  function setInformation(model, items) {
    const decision = model.decision_nodes[0];
    if (!decision) return;
    decision.information = model.chance_nodes.map((node) => node.name).filter((item) => items.includes(item));
    for (const key of Object.keys(decision.observation_costs))
      if (!decision.information.includes(key)) delete decision.observation_costs[key];
  }

  const parseNumber = (value) => {
    const text = String(value).trim();
    const number = Number(text);
    return text !== "" && Number.isFinite(number) ? number : text;
  };

  function setProbability(model, name, rowIndex, state, value) {
    const { node } = find(model, name);
    const row = node?.cpt?.[rowIndex];
    if (row) {
      row.probabilities[state] = parseNumber(value);
      filledRows.delete(row);
    }
    return row ? rowSum(row, node.states) : NaN;
  }

  function setUtility(model, name, rowIndex, value) {
    const { node } = find(model, name);
    const row = node?.table?.[rowIndex];
    if (row) {
      row.value = parseNumber(value);
      filledRows.delete(row);
    }
  }

  function setCost(model, variable, value) {
    const decision = model.decision_nodes[0];
    if (!decision) return;
    const parsed = parseNumber(value);
    if (parsed === "" || parsed === 0) delete decision.observation_costs[variable];
    else decision.observation_costs[variable] = parsed;
  }

  function normalizeTable(model, name, rowIndex = null) {
    const { node, kind } = find(model, name);
    if (kind !== "chance") return;
    node.cpt.forEach((row, i) => {
      if (rowIndex === null || i === rowIndex) {
        normalizeRow(node, row);
        filledRows.delete(row);
      }
    });
  }

  const rowOk = (row, states) => {
    const values = states.map((state) => row.probabilities[state]);
    return (
      values.every((v) => typeof v === "number" && v >= 0 && v <= 1) &&
      Math.abs(values.reduce((a, b) => a + b, 0) - 1) <= 1e-9
    );
  };

  /** Short warnings for the node list; the Python engine remains the full validator. */
  function problems(model, name) {
    const { node, kind } = find(model, name);
    const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
    const filled = (rows) => {
      const count = rows.filter((row) => filledRows.has(row)).length;
      return count ? [`${plural(count, "row")} filled with defaults; review ${count === 1 ? "it" : "them"}`] : [];
    };
    if (kind === "chance") {
      const bad = node.cpt.filter((row) => !rowOk(row, node.states)).length;
      return [...(bad ? [`${plural(bad, "row")} not adding up to 1`] : []), ...filled(node.cpt)];
    }
    if (kind === "utility") {
      const bad = node.table.filter((row) => typeof row.value !== "number").length;
      return [...(bad ? [`${plural(bad, "missing value")}`] : []), ...filled(node.table)];
    }
    if (kind === "decision" && !model.utility_nodes.length) return ["needs a utility node"];
    return [];
  }

  const formatSum = (sum) => (Number.isFinite(sum) ? Number(sum.toPrecision(10)).toString() : "—");
  const sumClass = (sum) => (Number.isFinite(sum) && Math.abs(sum - 1) <= 1e-9 ? "ok" : "bad");

  // ---------- Rendering ----------

  const FILLED =
    ' class="filled" title="This row was missing from the imported model and was filled with default values. ' +
    'Edit or normalize it to mark it as reviewed."';

  function statesEditor(node) {
    const chips = node.states
      .map(
        (state, i) => `
        <span class="state-chip">
          <input aria-label="State ${i + 1} of ${esc(node.name)}" value="${esc(state)}" maxlength="${MAX_NAME}"
            data-bfield="state-name" data-name="${esc(node.name)}" data-state="${esc(state)}"
            data-focus="state:${esc(node.name)}:${i}">
          ${
            node.states.length > MIN_STATES
              ? `<button class="chip-remove" data-b="remove-state" data-name="${esc(node.name)}"
                  data-state="${esc(state)}" aria-label="Remove state ${esc(state)}">×</button>`
              : ""
          }
        </span>`,
      )
      .join("");
    const add =
      node.states.length < MAX_STATES
        ? `<button class="button text" data-b="add-state" data-name="${esc(node.name)}"
            data-focus="add-state:${esc(node.name)}">+ Add state</button>`
        : "";
    return `<div class="state-chips">${chips}${add}</div>`;
  }

  function parentsEditor(model, node, kind) {
    const options = parentOptions(model, node.name);
    const selected = kind === "decision" ? node.information : node.parents;
    if (!options.length) return '<p class="field-hint">Add another variable to connect it here.</p>';
    return `<div class="parent-options">${options
      .map(({ name, cycle }) => {
        const checked = selected.includes(name);
        const disabled = cycle && !checked;
        const title = disabled ? ` title="Not allowed: ${esc(name)} depends on ${esc(node.name)}, so this would create a cycle."` : "";
        return `<label class="check${disabled ? " disabled" : ""}"${title}>
          <input type="checkbox" data-bfield="${kind === "decision" ? "information" : "parent"}"
            data-name="${esc(node.name)}" data-parent="${esc(name)}" data-focus="parent:${esc(node.name)}:${esc(name)}"
            ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}>
          <span>${esc(name)}</span></label>`;
      })
      .join("")}</div>`;
  }

  function moreRowsNote(count) {
    return count > MAX_ROWS_SHOWN
      ? `<p class="field-hint">Showing the first ${MAX_ROWS_SHOWN} of ${count.toLocaleString()} rows. Edit the rest in the JSON view.</p>`
      : "";
  }

  function cptEditor(node) {
    const head =
      node.parents.map((parent) => `<th class="when-col">${esc(parent)}</th>`).join("") +
      node.states.map((state) => `<th>P(${esc(state)})</th>`).join("") +
      "<th>Sum</th><th></th>";
    const rows = node.cpt
      .slice(0, MAX_ROWS_SHOWN)
      .map((row, r) => {
        const when = node.parents.map((parent) => `<td class="when-col">${esc(row.when[parent])}</td>`).join("");
        const cells = node.states
          .map(
            (state) => `<td><input type="number" step="any" min="0" max="1" inputmode="decimal"
              aria-label="P(${esc(node.name)} = ${esc(state)}${node.parents.length ? " | " + esc(node.parents.map((p) => `${p} = ${row.when[p]}`).join(", ")) : ""})"
              value="${esc(row.probabilities[state])}" data-bprob data-name="${esc(node.name)}" data-row="${r}"
              data-state="${esc(state)}" data-focus="prob:${r}:${esc(state)}"></td>`,
          )
          .join("");
        const sum = rowSum(row, node.states);
        return `<tr${filledRows.has(row) ? FILLED : ""}>${when}${cells}<td class="row-sum ${sumClass(sum)}" data-sum-row="${r}">${formatSum(sum)}</td>
          <td><button class="button text small" data-b="normalize-row" data-name="${esc(node.name)}" data-row="${r}"
            title="Scale this row so it adds up to 1">Normalize</button></td></tr>`;
      })
      .join("");
    return `<div class="table-wrap builder-table-wrap" tabindex="0" role="region" aria-label="Probability table of ${esc(node.name)}">
        <table class="builder-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      ${moreRowsNote(node.cpt.length)}
      <button class="button text" data-b="normalize-all" data-name="${esc(node.name)}">Normalize every row</button>`;
  }

  function utilityEditor(node) {
    const head = node.parents.map((parent) => `<th class="when-col">${esc(parent)}</th>`).join("") + "<th>Utility</th>";
    const rows = node.table
      .slice(0, MAX_ROWS_SHOWN)
      .map((row, r) => {
        const when = node.parents.map((parent) => `<td class="when-col">${esc(row.when[parent])}</td>`).join("");
        const label = node.parents.map((p) => `${p} = ${row.when[p]}`).join(", ") || "constant";
        return `<tr${filledRows.has(row) ? FILLED : ""}>${when}<td><input type="number" step="any" inputmode="decimal" value="${esc(row.value)}"
          aria-label="${esc(node.name)} utility for ${esc(label)}" data-butil data-name="${esc(node.name)}" data-row="${r}"
          data-focus="util:${r}"></td></tr>`;
      })
      .join("");
    return `<div class="table-wrap builder-table-wrap" tabindex="0" role="region" aria-label="Utility table of ${esc(node.name)}">
      <table class="builder-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      ${moreRowsNote(node.table.length)}`;
  }

  function costEditor(node) {
    if (!node.information.length)
      return '<p class="field-hint">Choose what the decision observes to set what each observation costs.</p>';
    return `<div class="cost-inputs">${node.information
      .map(
        (name) => `<label><span>${esc(name)}</span><input type="number" step="any" min="0" inputmode="decimal"
          value="${esc(owns(node.observation_costs, name) ? node.observation_costs[name] : "")}" placeholder="0" aria-label="Cost of observing ${esc(name)}"
          data-bcost data-name="${esc(name)}" data-focus="cost:${esc(name)}"></label>`,
      )
      .join("")}</div>`;
  }

  function nodeForm(model, name) {
    const { node, kind } = find(model, name);
    if (!node)
      return `<div class="empty-state"><h3>Start your network.</h3>
        <p>Add a chance variable, then connect variables by choosing their parents.</p></div>`;
    const label = { chance: "Chance variable", decision: "Decision", utility: "Utility" }[kind];
    const issues = problems(model, name);
    const warning = issues.length
      ? `<p class="builder-warning" role="note">Needs attention: ${esc(issues.join("; "))}.</p>`
      : "";
    const head = `
      <div class="builder-form-head">
        <label class="builder-name"><span class="node-type">${label}</span>
          <input value="${esc(node.name)}" maxlength="${MAX_NAME}" aria-label="Name of ${esc(node.name)}"
            data-bfield="node-name" data-name="${esc(node.name)}" data-focus="node-name"></label>
        <button class="button text danger" data-b="delete" data-name="${esc(node.name)}">Delete ${label.toLowerCase()}</button>
      </div>${warning}`;
    const section = (title, hint, body) =>
      `<section class="builder-section"><h4>${title}<span>${hint}</span></h4>${body}</section>`;
    if (kind === "chance")
      return (
        head +
        section("States", "2 to 8 possible values", statesEditor(node)) +
        section("Parents", "Which variables does it depend on?", parentsEditor(model, node, kind)) +
        section(
          node.parents.length ? "Conditional probability table" : "Prior probabilities",
          "Each row must add up to 1",
          cptEditor(node),
        )
      );
    if (kind === "decision")
      return (
        head +
        section("Actions", "2 to 8 choices", statesEditor(node)) +
        section("Observed before deciding", "Information available when acting", parentsEditor(model, node, kind)) +
        section("Cost of observing", "Utility units; leave blank if free", costEditor(node))
      );
    return (
      head +
      section("Depends on", "Outcomes and actions that affect this score", parentsEditor(model, node, kind)) +
      section("Utility table", "Preference scores; higher is better", utilityEditor(node))
    );
  }

  function nodeList(model, selected) {
    const item = (node, kind) => {
      const issues = problems(model, node.name);
      const meta =
        kind === "utility"
          ? `${node.parents.length} input${node.parents.length === 1 ? "" : "s"}`
          : `${node.states.length} ${kind === "decision" ? "actions" : "states"}`;
      return `<button class="builder-node ${kind} ${node.name === selected ? "active" : ""}" data-b="select"
          data-name="${esc(node.name)}" ${node.name === selected ? 'aria-current="true"' : ""}
          data-focus="node:${esc(node.name)}">
          <i aria-hidden="true"></i><span><strong>${esc(node.name)}</strong><small>${meta}</small></span>
          ${issues.length ? `<em class="issue" title="${esc(issues.join("; "))}">!</em>` : ""}
        </button>`;
    };
    const group = (title, nodes, kind) =>
      nodes.length
        ? `<div class="builder-group"><div class="builder-group-label">${title}</div>${nodes.map((n) => item(n, kind)).join("")}</div>`
        : "";
    return (
      group("Chance variables", model.chance_nodes, "chance") +
      group("Decision", model.decision_nodes, "decision") +
      group("Utilities", model.utility_nodes, "utility")
    );
  }

  function render(model, selected) {
    const full = allNames(model).length >= MAX_NODES;
    const hasDecision = model.decision_nodes.length > 0;
    return `
      <div class="builder">
        <div class="builder-toolbar">
          <label class="builder-model-name"><span>Model name</span>
            <input value="${esc(model.name)}" maxlength="${MAX_NAME}" data-bfield="model-name" data-focus="model-name"></label>
          <div class="builder-add" role="group" aria-label="Add a node">
            <button class="button secondary" data-b="add-chance" ${full ? "disabled" : ""}>+ Chance variable</button>
            <button class="button secondary" data-b="add-decision" ${full || hasDecision ? "disabled" : ""}
              ${hasDecision ? 'title="This version supports one decision"' : ""}>+ Decision</button>
            <button class="button secondary" data-b="add-utility" ${full ? "disabled" : ""}>+ Utility</button>
          </div>
        </div>
        <div class="builder-body">
          <nav class="builder-nodes" aria-label="Nodes in this network">${nodeList(model, selected)}</nav>
          <div class="builder-form">${nodeForm(model, selected)}</div>
        </div>
      </div>`;
  }

  return {
    MAX_NODES,
    normalize,
    find,
    allNames,
    addChance,
    addDecision,
    addUtility,
    removeNode,
    renameNode,
    setParents,
    setInformation,
    addState,
    removeState,
    renameState,
    setProbability,
    setUtility,
    setCost,
    normalizeTable,
    parentOptions,
    problems,
    render,
    formatSum,
    sumClass,
  };
})();
