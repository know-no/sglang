(function () {
  "use strict";

  var Course = window.SGLangCourse = window.SGLangCourse || {};

  function directWalkPanels(source) {
    return Array.from(source.querySelectorAll(":scope > [data-walk-panel]")).filter(function (node) {
      return (node.textContent || "").trim().length > 0;
    });
  }

  function phaseLabel(node, index) {
    if (node.dataset.walkLabel) return node.dataset.walkLabel;
    var text = (node.textContent || "").trim();
    var match = text.match(/(?:^|[。；;\s])(t\d+|第[一二三四五六七八九十\d]+轮|步骤\s*\d+|round\s*\d+)/i);
    if (match) return match[1].replace(/\s+/g, " ");
    return "阶段 " + (index + 1);
  }

  function codeCoordinates(node) {
    var values = [];
    node.querySelectorAll("code").forEach(function (code) {
      var value = (code.textContent || "").trim();
      if (value && value.length <= 42 && values.indexOf(value) === -1) values.push(value);
    });
    return values.slice(0, 6);
  }

  function initTeachingWalk(root, rootIndex) {
    var source = root.querySelector(".sf-teach-walk__source");
    if (!source) return;
    var panels = directWalkPanels(source);
    if (panels.length < 2) return;

    var rail = document.createElement("div");
    var toolbar = document.createElement("div");
    var status = document.createElement("span");
    var actions = document.createElement("div");
    var state = document.createElement("div");
    var tabs = [];
    var current = 0;

    rail.className = "teach-rail";
    rail.setAttribute("role", "tablist");
    rail.setAttribute("aria-label", "连续过程步骤");
    toolbar.className = "teach-toolbar";
    actions.className = "teach-actions";
    state.className = "teach-state";
    status.className = "teach-toolbar__status";
    status.setAttribute("aria-live", "polite");

    panels.forEach(function (panel, index) {
      var tab = document.createElement("button");
      var panelId = "teach-walk-" + (rootIndex + 1) + "-panel-" + (index + 1);
      var tabId = "teach-walk-" + (rootIndex + 1) + "-tab-" + (index + 1);
      panel.id = panel.id || panelId;
      panel.dataset.walkPanel = String(index + 1);
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", tabId);
      tab.type = "button";
      tab.id = tabId;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", panel.id);
      tab.textContent = phaseLabel(panel, index);
      tab.addEventListener("click", function () {
        select(index);
      });
      tab.addEventListener("keydown", function (event) {
        var next = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          next = (index + 1) % panels.length;
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          next = (index - 1 + panels.length) % panels.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = panels.length - 1;
        } else {
          return;
        }
        event.preventDefault();
        select(next, true);
      });
      rail.appendChild(tab);
      tabs.push(tab);
    });

    [
      ["prev", "上一步"],
      ["next", "下一步"],
      ["all", "展开全部"]
    ].forEach(function (entry) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "teach-actions__button";
      button.dataset.teachAction = entry[0];
      button.textContent = entry[1];
      actions.appendChild(button);
    });

    function renderCoordinates(panel) {
      var values = codeCoordinates(panel);
      state.replaceChildren();
      var label = document.createElement("b");
      label.textContent = "本步对象 / 状态";
      state.appendChild(label);
      if (!values.length) {
        var fallback = document.createElement("span");
        fallback.textContent = "以本步加粗对象和状态写入为坐标";
        state.appendChild(fallback);
        return;
      }
      values.forEach(function (value) {
        var code = document.createElement("code");
        code.textContent = value;
        state.appendChild(code);
      });
    }

    function select(index, focus) {
      current = Math.max(0, Math.min(index, panels.length - 1));
      panels.forEach(function (panel, panelIndex) {
        panel.hidden = panelIndex !== current;
      });
      tabs.forEach(function (tab, tabIndex) {
        var active = tabIndex === current;
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
      });
      status.textContent = "步骤 " + (current + 1) + " / " + panels.length;
      renderCoordinates(panels[current]);
      if (focus) tabs[current].focus();
    }

    function showAll() {
      panels.forEach(function (panel) {
        panel.hidden = false;
      });
      tabs.forEach(function (tab, tabIndex) {
        tab.setAttribute("aria-selected", "false");
        tab.tabIndex = tabIndex === current ? 0 : -1;
      });
      state.replaceChildren();
      status.textContent = "全部 " + panels.length + " 个阶段已展开";
    }

    actions.addEventListener("click", function (event) {
      var button = event.target.closest("[data-teach-action]");
      if (!button) return;
      if (button.dataset.teachAction === "prev") {
        select(current - 1);
      } else if (button.dataset.teachAction === "next") {
        select(current + 1);
      } else {
        showAll();
      }
    });

    toolbar.append(status, actions);
    root.insertBefore(rail, source);
    source.appendChild(state);
    root.appendChild(toolbar);
    root.dataset.enhanced = "true";
    function revealAnchor() {
      var id;
      try { id = decodeURIComponent(location.hash.slice(1)); } catch (_error) { return; }
      var target = id && document.getElementById(id);
      if (!target || !root.contains(target)) return;
      var index = panels.findIndex(function (panel) { return panel === target || panel.contains(target); });
      if (index < 0) return;
      select(index);
      requestAnimationFrame(function () { target.scrollIntoView({ block: "start" }); });
    }
    select(0);
    revealAnchor();
    window.addEventListener("hashchange", revealAnchor);
  }

  function normalize(value) {
    return (value || "")
      .replace(/<br\s*\/?\s*>/gi, " ")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
  }

  function initObjectMap(root) {
    var topic = root.closest(".sf-topic--objects");
    if (!topic) return;
    var body = topic.querySelector(".sf-topic__body");
    if (!body) return;
    var status = root.querySelector("[data-object-status]");
    var controls = Array.from(root.querySelectorAll("[data-object-key]"));

    function locate(key) {
      var normalizedKey = normalize(key);
      var candidates = Array.from(body.querySelectorAll([
        "code",
        "td",
        "th",
        "p",
        "li",
        "pre.mermaid",
        ".diagram-frame svg text",
        ".diagram-frame"
      ].join(","))).filter(function (node) {
        return !root.contains(node);
      });
      var exact = candidates.filter(function (node) {
        return normalize(node.textContent) === normalizedKey;
      });
      var rawMatches = exact.length ? exact : candidates.filter(function (node) {
        var text = normalize(node.textContent);
        return text.indexOf(normalizedKey) !== -1 ||
          text.replace(/\s+/g, "").indexOf(normalizedKey.replace(/\s+/g, "")) !== -1;
      });
      return rawMatches.reduce(function (matches, node) {
        var target = node.closest("td, th, p, li, figure") || node;
        if (matches.indexOf(target) === -1) matches.push(target);
        return matches;
      }, []);
    }

    function clearHits() {
      body.querySelectorAll(".is-object-hit").forEach(function (node) {
        node.classList.remove("is-object-hit");
      });
      controls.forEach(function (control) {
        control.setAttribute("aria-pressed", "false");
      });
    }

    function activate(control) {
      var key = control.dataset.objectKey || "";
      var matches = locate(key);
      clearHits();
      matches.forEach(function (node) {
        node.classList.add("is-object-hit");
      });
      control.setAttribute("aria-pressed", "true");
      if (status) status.textContent = "正文中找到 " + matches.length + " 处 “" + key + "”";
      if (matches[0]) {
        var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        matches[0].scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
      }
    }

    controls.forEach(function (control) {
      control.setAttribute("aria-pressed", "false");
      if (control.tagName !== "BUTTON") {
        control.setAttribute("role", "button");
        control.tabIndex = 0;
        control.addEventListener("keydown", function (event) {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          activate(control);
        });
      }
      control.addEventListener("click", function () {
        activate(control);
      });
    });
  }

  function cellHtml(cell) {
    return cell ? cell.innerHTML : "";
  }

  function initStateFlow(root, rootIndex) {
    var table = root.querySelector("table");
    if (!table) return;
    var rows = Array.from(table.querySelectorAll("tbody tr")).filter(function (row) {
      return row.cells && row.cells.length >= 3;
    });
    if (!rows.length) return;

    var visual = document.createElement("div");
    var rail = document.createElement("ol");
    var reason = document.createElement("div");
    var toolbar = document.createElement("div");
    var status = document.createElement("span");
    var actions = document.createElement("div");
    var tabs = [];
    var current = 0;

    visual.className = "state-flow";
    rail.className = "state-flow__rail";
    rail.setAttribute("role", "tablist");
    rail.setAttribute("aria-label", "源码动作与状态变化");
    reason.className = "state-flow__reason";
    reason.setAttribute("role", "tabpanel");
    reason.setAttribute("aria-live", "polite");
    toolbar.className = "state-flow__toolbar";
    status.className = "state-flow__status";
    status.setAttribute("aria-live", "polite");
    actions.className = "state-flow__actions";

    rows.forEach(function (row, index) {
      var item = document.createElement("li");
      var tab = document.createElement("button");
      var action = document.createElement("span");
      var arrow = document.createElement("span");
      var changed = document.createElement("span");
      var tabId = "state-flow-" + (rootIndex + 1) + "-tab-" + (index + 1);
      tab.type = "button";
      tab.id = tabId;
      tab.className = "state-flow__step";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", "state-flow-" + (rootIndex + 1) + "-reason");
      action.className = "state-flow__action";
      changed.className = "state-flow__changed";
      action.innerHTML = cellHtml(row.cells[0]);
      changed.innerHTML = cellHtml(row.cells[1]);
      arrow.className = "state-flow__arrow";
      arrow.textContent = "→";
      tab.append(action, arrow, changed);
      tab.addEventListener("click", function () {
        select(index);
      });
      tab.addEventListener("keydown", function (event) {
        var next = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          next = (index + 1) % rows.length;
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          next = (index - 1 + rows.length) % rows.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = rows.length - 1;
        } else {
          return;
        }
        event.preventDefault();
        select(next, true);
      });
      item.appendChild(tab);
      rail.appendChild(item);
      tabs.push(tab);
    });

    reason.id = "state-flow-" + (rootIndex + 1) + "-reason";
    [
      ["prev", "上一项"],
      ["next", "下一项"]
    ].forEach(function (entry) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "state-flow__button";
      button.dataset.stateAction = entry[0];
      button.textContent = entry[1];
      actions.appendChild(button);
    });

    function select(index, focus) {
      current = Math.max(0, Math.min(index, rows.length - 1));
      tabs.forEach(function (tab, tabIndex) {
        var active = tabIndex === current;
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
      });
      reason.setAttribute("aria-labelledby", tabs[current].id);
      reason.innerHTML = "<b>为什么：</b> " + cellHtml(rows[current].cells[2]);
      status.textContent = "状态变化 " + (current + 1) + " / " + rows.length;
      if (focus) tabs[current].focus();
    }

    actions.addEventListener("click", function (event) {
      var button = event.target.closest("[data-state-action]");
      if (!button) return;
      select(button.dataset.stateAction === "next" ? current + 1 : current - 1);
    });

    toolbar.append(status, actions);
    visual.append(rail, reason, toolbar);
    table.hidden = true;
    table.dataset.stateFlowFallback = "true";
    var tableWrap = table.closest(".teaching-table");
    if (tableWrap) {
      tableWrap.hidden = true;
      tableWrap.dataset.stateFlowFallback = "true";
    }
    root.insertBefore(visual, root.firstChild);
    root.dataset.enhanced = "true";
    select(0);
  }

  function initStageMapSpy() {
    var links = Array.from(document.querySelectorAll(".sf-stage-map a[href^='#']"));
    if (!links.length || !("IntersectionObserver" in window)) return;
    var byId = new Map();
    links.forEach(function (link) {
      try {
        byId.set(decodeURIComponent(link.hash.slice(1)), link);
      } catch (_error) {
        byId.set(link.hash.slice(1), link);
      }
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (link) {
          link.removeAttribute("aria-current");
        });
        var link = byId.get(entry.target.id);
        if (link) link.setAttribute("aria-current", "step");
      });
    }, { rootMargin: "-20% 0px -72% 0px", threshold: 0 });
    byId.forEach(function (_link, id) {
      var topic = document.getElementById(id);
      if (topic) observer.observe(topic);
    });
  }

  function initDiagramAtlas() {
    var lesson = document.querySelector("[data-lesson]");
    if (!lesson) return;
    var figures = Array.from(lesson.querySelectorAll("[data-diagram-frame]"));
    if (figures.length < 2) return;
    var kindLabels = {
      architecture: "架构边界",
      component: "组件所有权",
      sequence: "时序泳道",
      state: "状态迁移",
      data: "数据 / Tensor",
      code: "源码流程",
      generic: "关系图"
    };
    var atlas = document.createElement("nav");
    var head = document.createElement("div");
    var list = document.createElement("ol");
    var kindCounts = new Map();
    atlas.className = "sf-diagram-atlas";
    atlas.setAttribute("aria-label", "本章互补图谱");
    head.className = "sf-diagram-atlas__head";
    list.className = "sf-diagram-atlas__list";

    figures.forEach(function (figure, index) {
      var kind = figure.dataset.diagramKind || "generic";
      var topic = figure.closest(".sf-topic");
      var topicTitle = topic && topic.querySelector(".sf-sec-title");
      var item = document.createElement("li");
      var link = document.createElement("a");
      var badge = document.createElement("b");
      var label = document.createElement("span");
      figure.id = figure.id || "chapter-diagram-" + (index + 1);
      kindCounts.set(kind, (kindCounts.get(kind) || 0) + 1);
      item.dataset.diagramKind = kind;
      link.href = "#" + figure.id;
      badge.textContent = kindLabels[kind] || kindLabels.generic;
      label.textContent = topicTitle ? topicTitle.textContent.replace(/^\s*\d+\s*/, "").trim() : "图 " + (index + 1);
      link.append(badge, label);
      item.appendChild(link);
      list.appendChild(item);
    });

    var summary = Array.from(kindCounts.entries()).map(function (entry) {
      return (kindLabels[entry[0]] || kindLabels.generic) + " " + entry[1];
    }).join(" · ");
    head.innerHTML = "<span>Visual lenses · 本章 " + figures.length + " 张互补图</span><b>" + summary + "</b>";
    atlas.append(head, list);
    lesson.parentNode.insertBefore(atlas, lesson);
  }

  document.querySelectorAll("[data-teaching-walk]").forEach(initTeachingWalk);
  document.querySelectorAll("[data-object-map]").forEach(initObjectMap);
  document.querySelectorAll("[data-state-flow]").forEach(initStateFlow);
  initDiagramAtlas();
  initStageMapSpy();

  Course.lessonComponents = {
    teachingWalks: document.querySelectorAll("[data-teaching-walk]").length,
    objectMaps: document.querySelectorAll("[data-object-map]").length,
    stateFlows: document.querySelectorAll("[data-state-flow]").length,
    selfTests: document.querySelectorAll("[data-self-test]").length,
    diagramAtlases: document.querySelectorAll(".sf-diagram-atlas").length
  };
})();
