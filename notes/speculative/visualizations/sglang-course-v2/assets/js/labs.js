(function () {
  "use strict";

  var Course = window.SGLangCourse = window.SGLangCourse || {};
  var stepLabs = [];
  var variantGroups = [];

  function readQuery(key) {
    try { return new URL(location.href).searchParams.get(key); }
    catch (_error) { return null; }
  }

  function writeQuery(key, value) {
    try {
      var url = new URL(location.href);
      url.searchParams.set(key, value);
      history.replaceState(null, "", url.pathname.split("/").pop() + url.search + url.hash);
    } catch (_error) {
      /* State remains usable without a shareable URL. */
    }
  }

  function initStepLab(root, rootIndex) {
    var buttons = Array.from(root.querySelectorAll("[data-lab-step]"));
    var panels = Array.from(root.querySelectorAll("[data-lab-panel]"));
    if (!buttons.length || !panels.length) return;

    var id = root.dataset.labId || root.id || ("lab-" + (rootIndex + 1));
    var key = root.dataset.labKey || ("lab-" + id);
    var values = buttons.map(function (button, index) { return button.dataset.labStep || String(index + 1); });
    var initial = root.dataset.defaultStep || values[0];
    var requested = readQuery(key);
    var current = values.indexOf(requested) !== -1 ? requested : initial;
    var status = root.querySelector("[data-lab-status]");
    var tablist = root.querySelector("[data-lab-tabs]") || buttons[0].parentElement;
    if (tablist) tablist.setAttribute("role", "tablist");

    root.dataset.enhanced = "true";
    buttons.forEach(function (button, index) {
      var panel = panels.find(function (candidate) { return candidate.dataset.labPanel === values[index]; }) || panels[index];
      if (!button.id) button.id = id + "-step-" + (index + 1);
      if (panel && !panel.id) panel.id = id + "-panel-" + (index + 1);
      button.type = "button";
      button.setAttribute("role", "tab");
      if (panel) {
        button.setAttribute("aria-controls", panel.id);
        panel.setAttribute("role", "tabpanel");
        panel.setAttribute("aria-labelledby", button.id);
      }
    });

    function select(value, options) {
      var next = values.indexOf(value) !== -1 ? value : values[0];
      var activeIndex = values.indexOf(next);
      current = next;

      buttons.forEach(function (button, index) {
        var active = index === activeIndex;
        button.setAttribute("aria-selected", String(active));
        button.tabIndex = active ? 0 : -1;
        button.classList.toggle("is-active", active);
      });
      panels.forEach(function (panel, index) {
        var panelValue = panel.dataset.labPanel || values[index];
        panel.hidden = panelValue !== next;
      });
      root.querySelectorAll("[data-lab-steps]").forEach(function (node) {
        var allowed = (node.dataset.labSteps || "").split(/\s+/).filter(Boolean);
        node.dataset.active = String(allowed.indexOf(next) !== -1);
      });
      if (status) status.textContent = "步骤 " + (activeIndex + 1) + " / " + values.length;
      if (!options || options.syncUrl !== false) writeQuery(key, next);
      if (options && options.focus) buttons[activeIndex].focus();
      root.dispatchEvent(new CustomEvent("sglang:lab-step", {
        bubbles: true,
        detail: { id: id, key: key, value: next, index: activeIndex, total: values.length }
      }));
    }

    buttons.forEach(function (button, index) {
      button.addEventListener("click", function () { select(values[index]); });
      button.addEventListener("keydown", function (event) {
        var nextIndex = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % buttons.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + buttons.length) % buttons.length;
        else if (event.key === "Home") nextIndex = 0;
        else if (event.key === "End") nextIndex = buttons.length - 1;
        else return;
        event.preventDefault();
        select(values[nextIndex], { focus: true });
      });
    });

    root.addEventListener("click", function (event) {
      var action = event.target.closest("[data-lab-action]");
      if (!action || !root.contains(action)) return;
      var index = values.indexOf(current);
      if (action.dataset.labAction === "next") select(values[Math.min(index + 1, values.length - 1)]);
      else if (action.dataset.labAction === "prev") select(values[Math.max(index - 1, 0)]);
      else if (action.dataset.labAction === "reset") select(initial);
    });

    select(current, { syncUrl: Boolean(requested) });
    stepLabs.push({ root: root, select: select, get value() { return current; } });
  }

  function initVariantGroup(root, rootIndex) {
    var buttons = Array.from(root.querySelectorAll("[data-variant]"));
    if (!buttons.length) return;
    var id = root.dataset.variantId || root.id || ("variant-" + (rootIndex + 1));
    var key = root.dataset.variantKey || ("variant-" + id);
    var values = buttons.map(function (button) { return button.dataset.variant; });
    var initial = root.dataset.defaultVariant || values[0];
    var requested = readQuery(key);
    var current = values.indexOf(requested) !== -1 ? requested : initial;

    root.dataset.enhanced = "true";
    buttons.forEach(function (button) {
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
    });

    function select(value, options) {
      var next = values.indexOf(value) !== -1 ? value : values[0];
      current = next;
      buttons.forEach(function (button) {
        var active = button.dataset.variant === next;
        button.setAttribute("aria-pressed", String(active));
        button.classList.toggle("is-active", active);
      });
      root.querySelectorAll("[data-variants]").forEach(function (node) {
        var allowed = (node.dataset.variants || "").split(/\s+/).filter(Boolean);
        node.dataset.active = String(allowed.indexOf(next) !== -1);
      });
      if (!options || options.syncUrl !== false) writeQuery(key, next);
      root.dispatchEvent(new CustomEvent("sglang:variant", {
        bubbles: true,
        detail: { id: id, key: key, value: next }
      }));
    }

    buttons.forEach(function (button, index) {
      button.addEventListener("click", function () { select(values[index]); });
      button.addEventListener("keydown", function (event) {
        var nextIndex = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % buttons.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + buttons.length) % buttons.length;
        else return;
        event.preventDefault();
        buttons[nextIndex].focus();
        select(values[nextIndex]);
      });
    });

    select(current, { syncUrl: Boolean(requested) });
    variantGroups.push({ root: root, select: select, get value() { return current; } });
  }

  function initBoundInputs() {
    document.querySelectorAll("[data-lab-input]").forEach(function (input, index) {
      if (!input.id) input.id = "lab-input-" + (index + 1);
      var outputs = Array.from(document.querySelectorAll("[data-lab-output=\"" + CSS.escape(input.id) + "\"]"));

      function render() {
        var suffix = input.dataset.unit || "";
        outputs.forEach(function (output) {
          output.value = input.value + suffix;
          output.textContent = input.value + suffix;
        });
        input.dispatchEvent(new CustomEvent("sglang:lab-input", {
          bubbles: true,
          detail: { id: input.id, value: input.value, valueAsNumber: input.valueAsNumber }
        }));
      }

      input.addEventListener("input", render);
      input.addEventListener("change", render);
      render();
    });
  }

  document.querySelectorAll("[data-step-lab]").forEach(initStepLab);
  document.querySelectorAll("[data-variant-group]").forEach(initVariantGroup);
  initBoundInputs();

  function revealAnchor() {
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_error) { return; }
    var target = id && document.getElementById(id);
    if (!target) return;
    var revealed = false;
    stepLabs.forEach(function (lab) {
      var panel = target.closest("[data-lab-panel]");
      if (panel && lab.root.contains(panel)) {
        lab.select(panel.dataset.labPanel, { syncUrl: false });
        revealed = true;
      }
    });
    if (revealed) requestAnimationFrame(function () { target.scrollIntoView({ block: "start" }); });
  }
  Course.labs = { steps: stepLabs, variants: variantGroups };
  revealAnchor();
  window.addEventListener("hashchange", revealAnchor);
})();
