(function () {
  "use strict";

  var Course = window.SGLangCourse = window.SGLangCourse || {};
  var instances = [];

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
      /* The interaction still works when URL mutation is unavailable. */
    }
  }

  function init(root, rootIndex) {
    var buttons = Array.from(root.querySelectorAll("[data-source-step]"));
    var panels = Array.from(root.querySelectorAll("[data-source-panel]"));
    if (!buttons.length || !panels.length) return;

    var key = root.dataset.sourceKey || ("source-step-" + (rootIndex + 1));
    var values = buttons.map(function (button, index) {
      return button.dataset.sourceStep || String(index + 1);
    });
    var defaultValue = root.dataset.defaultStep || values[0];
    var requested = readQuery(key);
    var current = values.indexOf(requested) !== -1 ? requested : defaultValue;
    var status = root.querySelector("[data-source-status]");
    var tabs = root.querySelector("[data-source-tabs]") || buttons[0].parentElement;

    if (tabs) tabs.setAttribute("role", "tablist");
    root.dataset.enhanced = "true";

    buttons.forEach(function (button, index) {
      var value = values[index];
      var panel = panels.find(function (candidate) { return candidate.dataset.sourcePanel === value; }) || panels[index];
      if (!button.id) button.id = (root.id || "source-microscope-" + (rootIndex + 1)) + "-tab-" + (index + 1);
      if (panel && !panel.id) panel.id = (root.id || "source-microscope-" + (rootIndex + 1)) + "-panel-" + (index + 1);
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
      current = next;
      var activeIndex = values.indexOf(next);

      buttons.forEach(function (button, index) {
        var active = values[index] === next;
        button.setAttribute("aria-selected", String(active));
        button.tabIndex = active ? 0 : -1;
        button.classList.toggle("is-active", active);
      });

      panels.forEach(function (panel, index) {
        var panelValue = panel.dataset.sourcePanel || values[index];
        panel.hidden = panelValue !== next;
      });

      root.querySelectorAll("[data-source-steps]").forEach(function (node) {
        var allowed = (node.dataset.sourceSteps || "").split(/\s+/).filter(Boolean);
        node.dataset.active = String(allowed.indexOf(next) !== -1);
      });

      if (status) status.textContent = "步骤 " + (activeIndex + 1) + " / " + values.length;
      if (!options || options.syncUrl !== false) writeQuery(key, next);
      if (options && options.focus) buttons[activeIndex].focus();

      root.dispatchEvent(new CustomEvent("sglang:source-step", {
        bubbles: true,
        detail: { key: key, value: next, index: activeIndex, total: values.length }
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
      var action = event.target.closest("[data-source-action]");
      if (!action || !root.contains(action)) return;
      var index = values.indexOf(current);
      if (action.dataset.sourceAction === "next") select(values[Math.min(index + 1, values.length - 1)]);
      else if (action.dataset.sourceAction === "prev") select(values[Math.max(index - 1, 0)]);
      else if (action.dataset.sourceAction === "reset") select(defaultValue);
    });

    select(current, { syncUrl: Boolean(requested) });
    instances.push({ root: root, select: select, get value() { return current; } });
  }

  document.querySelectorAll("[data-source-microscope]").forEach(init);
  Course.sourceMicroscopes = instances;
})();
