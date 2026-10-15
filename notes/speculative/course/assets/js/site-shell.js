(function () {
  "use strict";

  document.documentElement.classList.add("js");

  var Course = window.SGLangCourse = window.SGLangCourse || {};
  var STORAGE_PREFIX = "sglang-course-v2:";
  var STORAGE_KEYS = { theme: true, "nav-pinned": true };
  var themeModes = ["dark", "light", "system"];
  var memoryStore = Object.create(null);

  function readStorage(key, fallback) {
    if (!STORAGE_KEYS[key]) return fallback;
    try {
      var value = window.localStorage.getItem(STORAGE_PREFIX + key);
      return value === null ? fallback : value;
    } catch (_error) {
      return Object.prototype.hasOwnProperty.call(memoryStore, key) ? memoryStore[key] : fallback;
    }
  }

  function writeStorage(key, value) {
    if (!STORAGE_KEYS[key]) return false;
    memoryStore[key] = value;
    try {
      window.localStorage.setItem(STORAGE_PREFIX + key, value);
      return true;
    } catch (_error) {
      return false;
    }
  }

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  function applyTheme(mode) {
    var normalized = themeModes.indexOf(mode) === -1 ? "dark" : mode;
    var resolved = normalized === "system" ? systemTheme() : normalized;
    var labels = { dark: "深色", light: "浅色", system: "跟随系统" };

    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themeMode = normalized;
    document.documentElement.style.colorScheme = resolved;

    document.querySelectorAll("button[data-theme-toggle]").forEach(function (button) {
      button.setAttribute("aria-label", "当前主题：" + labels[normalized] + "。点击切换主题");
      button.dataset.themeMode = normalized;
      var label = button.querySelector("[data-theme-label]");
      if (label) label.textContent = labels[normalized];
    });

    window.dispatchEvent(new CustomEvent("sglang:theme", {
      detail: { mode: normalized, resolved: resolved }
    }));
  }

  function initTheme() {
    applyTheme(readStorage("theme", "dark"));

    document.addEventListener("click", function (event) {
      var button = event.target.closest("button[data-theme-toggle]");
      if (!button) return;
      var current = document.documentElement.dataset.themeMode || "dark";
      var next = themeModes[(themeModes.indexOf(current) + 1) % themeModes.length];
      writeStorage("theme", next);
      applyTheme(next);
    });

    if (window.matchMedia) {
      var media = window.matchMedia("(prefers-color-scheme: light)");
      var onChange = function () {
        if (document.documentElement.dataset.themeMode === "system") applyTheme("system");
      };
      if (media.addEventListener) media.addEventListener("change", onChange);
      else if (media.addListener) media.addListener(onChange);
    }
  }

  function initNavigationDrawer() {
    var body = document.body;
    var drawer = document.getElementById("course-navigation");
    var mobile = window.matchMedia ? window.matchMedia("(max-width: 63.999rem)") : null;
    var pinEligible = window.matchMedia ? window.matchMedia("(min-width: 64rem)") : null;
    var pinPreference = readStorage("nav-pinned", "false") === "true";
    var opener = null;

    body.dataset.navOpen = "false";
    body.dataset.navPinned = "false";
    if (!drawer) return;

    function canPin() {
      return Boolean(pinEligible && pinEligible.matches);
    }

    function isOpen() {
      return body.dataset.navOpen === "true";
    }

    function isPinned() {
      return body.dataset.navPinned === "true";
    }

    function isVisible() {
      return isOpen() || isPinned();
    }

    function setInteractive(interactive) {
      drawer.inert = !interactive;
      if (interactive) {
        drawer.removeAttribute("inert");
        drawer.removeAttribute("aria-hidden");
      } else {
        drawer.setAttribute("inert", "");
        drawer.setAttribute("aria-hidden", "true");
      }
    }

    function updateControls() {
      var visible = isVisible();
      var eligible = canPin();

      document.querySelectorAll("button[data-nav-toggle]").forEach(function (button) {
        button.setAttribute("aria-controls", "course-navigation");
        button.setAttribute("aria-expanded", String(visible));
        button.setAttribute("aria-label", visible ? "收起课程目录" : "打开课程目录");
      });

      document.querySelectorAll("button[data-nav-pin]").forEach(function (button) {
        button.disabled = !eligible;
        button.setAttribute("aria-disabled", String(!eligible));
        button.setAttribute("aria-pressed", String(isPinned()));
        button.setAttribute("aria-label", isPinned() ? "取消固定课程目录" : "固定课程目录");
        button.title = eligible
          ? (isPinned() ? "取消固定课程目录" : "固定课程目录")
          : "窄屏使用覆盖式目录；宽度达到 64rem 后可固定";
        var label = button.querySelector("[data-nav-pin-label]");
        if (label) label.textContent = isPinned() ? "取消固定" : "固定目录";
      });

      setInteractive(visible);
    }

    function firstDrawerControl() {
      return drawer.querySelector("button[data-nav-close]") || drawer.querySelector(
        "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), " +
        "textarea:not([disabled]), summary, [tabindex]:not([tabindex=\"-1\"])"
      );
    }

    function restoreOpener() {
      var fallback = document.querySelector("button[data-nav-toggle]");
      var target = opener && document.documentElement.contains(opener) ? opener : fallback;
      if (target && typeof target.focus === "function") target.focus();
      opener = null;
    }

    function setOpen(open, options) {
      var settings = options || {};
      var wasOpen = isOpen();
      var next = Boolean(open) && !isPinned();

      if (next && !wasOpen) opener = settings.opener || document.activeElement;
      body.dataset.navOpen = String(next);
      updateControls();

      if (next) {
        window.requestAnimationFrame(function () {
          if (!isOpen()) return;
          var first = firstDrawerControl();
          if (first) first.focus();
        });
      } else if (wasOpen && settings.restoreFocus !== false) {
        restoreOpener();
      }
    }

    function setPinned(pinned, options) {
      var settings = options || {};
      var next = Boolean(pinned) && canPin();
      var wasVisible = isVisible();

      if (settings.persist !== false) {
        pinPreference = next;
        writeStorage("nav-pinned", String(next));
      }
      body.dataset.navPinned = String(next);
      body.dataset.navOpen = "false";
      updateControls();

      if (!next && wasVisible && settings.restoreFocus !== false) restoreOpener();
    }

    function closeDrawer(options) {
      if (isPinned()) setPinned(false, options);
      else setOpen(false, options);
    }

    function syncViewport() {
      var activeWasInside = drawer.contains(document.activeElement);
      var shouldPin = canPin() && pinPreference;

      body.dataset.navPinned = String(shouldPin);
      body.dataset.navOpen = "false";
      updateControls();

      if (!shouldPin && activeWasInside) restoreOpener();
    }

    document.addEventListener("click", function (event) {
      var toggle = event.target.closest("button[data-nav-toggle]");
      if (toggle) {
        if (isPinned()) setPinned(false, { restoreFocus: false });
        else setOpen(!isOpen(), { opener: toggle });
        return;
      }

      var pin = event.target.closest("button[data-nav-pin]");
      if (pin) {
        if (!canPin()) return;
        if (!opener) opener = document.querySelector("button[data-nav-toggle]");
        var nextPinned = !isPinned();
        setPinned(nextPinned, { restoreFocus: !nextPinned });
        return;
      }

      if (event.target.closest("button[data-nav-close], button[data-nav-backdrop]")) {
        closeDrawer();
        return;
      }

      if (isVisible() && event.target.closest("#course-navigation a")) {
        if (!isPinned()) setOpen(false, { restoreFocus: false });
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && isOpen()) {
        event.preventDefault();
        setOpen(false);
        return;
      }

      if (event.key !== "Tab" || !isOpen() || !(mobile && mobile.matches)) return;
      var focusable = Array.from(drawer.querySelectorAll(
        "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), " +
        "textarea:not([disabled]), summary, [tabindex]:not([tabindex=\"-1\"])"
      )).filter(function (element) {
        return element.getAttribute("aria-hidden") !== "true";
      });
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });

    var onViewportChange = function () { syncViewport(); };
    if (mobile) {
      if (mobile.addEventListener) mobile.addEventListener("change", onViewportChange);
      else if (mobile.addListener) mobile.addListener(onViewportChange);
    }
    if (pinEligible) {
      if (pinEligible.addEventListener) pinEligible.addEventListener("change", onViewportChange);
      else if (pinEligible.addListener) pinEligible.addListener(onViewportChange);
    }

    syncViewport();
  }

  function initCurrentPage() {
    var currentId = document.body.dataset.chapterId || document.body.dataset.pageId || "";
    if (!currentId) return;
    document.querySelectorAll("[data-course-link][data-chapter-id]").forEach(function (link) {
      if (link.dataset.chapterId === currentId) {
        link.setAttribute("aria-current", "page");
        var parent = link.closest("details");
        while (parent) {
          parent.open = true;
          parent = parent.parentElement ? parent.parentElement.closest("details") : null;
        }
      } else if (link.getAttribute("aria-current") === "page") {
        link.removeAttribute("aria-current");
      }
    });
  }

  function initExternalLinks() {
    document.querySelectorAll("a[target=\"_blank\"]").forEach(function (link) {
      var rel = new Set((link.getAttribute("rel") || "").split(/\s+/).filter(Boolean));
      rel.add("noopener");
      rel.add("noreferrer");
      link.setAttribute("rel", Array.from(rel).join(" "));
    });
  }

  function mermaidThemeVariables(resolvedTheme) {
    if (resolvedTheme === "light") {
      return {
        fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
        fontSize: "14px",
        background: "#ffffff",
        primaryColor: "#f6f8fa",
        primaryTextColor: "#1f2328",
        primaryBorderColor: "#0969da",
        secondaryColor: "#f3f0ff",
        secondaryTextColor: "#24292f",
        secondaryBorderColor: "#8250df",
        tertiaryColor: "#dafbe1",
        tertiaryTextColor: "#1f2328",
        tertiaryBorderColor: "#1a7f37",
        lineColor: "#57606a",
        textColor: "#1f2328",
        mainBkg: "#f6f8fa",
        nodeBorder: "#0969da",
        clusterBkg: "#ffffff",
        clusterBorder: "#d0d7de",
        edgeLabelBackground: "#ffffff",
        actorBkg: "#f6f8fa",
        actorBorder: "#0969da",
        actorTextColor: "#1f2328",
        actorLineColor: "#8c959f",
        signalColor: "#0969da",
        signalTextColor: "#1f2328",
        labelBoxBkgColor: "#f6f8fa",
        labelBoxBorderColor: "#0969da",
        labelTextColor: "#1f2328",
        loopTextColor: "#1f2328",
        noteBkgColor: "#fff8c5",
        noteBorderColor: "#9a6700",
        noteTextColor: "#3b2300",
        activationBkgColor: "#f3f0ff",
        activationBorderColor: "#8250df"
      };
    }

    return {
      fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
      fontSize: "14px",
      background: "#0d1117",
      primaryColor: "#161b22",
      primaryTextColor: "#e6edf3",
      primaryBorderColor: "#58a6ff",
      secondaryColor: "#1c2330",
      secondaryTextColor: "#e6edf3",
      secondaryBorderColor: "#bc8cff",
      tertiaryColor: "#10251a",
      tertiaryTextColor: "#e6edf3",
      tertiaryBorderColor: "#3fb950",
      lineColor: "#8b949e",
      textColor: "#e6edf3",
      mainBkg: "#161b22",
      nodeBorder: "#58a6ff",
      clusterBkg: "#161b22",
      clusterBorder: "#30363d",
      edgeLabelBackground: "#0d1117",
      actorBkg: "#161b22",
      actorBorder: "#58a6ff",
      actorTextColor: "#e6edf3",
      actorLineColor: "#484f58",
      signalColor: "#58a6ff",
      signalTextColor: "#e6edf3",
      labelBoxBkgColor: "#1c2330",
      labelBoxBorderColor: "#58a6ff",
      labelTextColor: "#e6edf3",
      loopTextColor: "#e6edf3",
      noteBkgColor: "#3b2e0b",
      noteBorderColor: "#e3b341",
      noteTextColor: "#f0e6c2",
      activationBkgColor: "#2b1f3d",
      activationBorderColor: "#bc8cff"
    };
  }

  var MERMAID_BASE_FONT_PX = 14;
  var MERMAID_MIN_EFFECTIVE_FONT_PX = 12;

  function mermaidTone(label, kind, index, element) {
    var text = String(label || "").toLowerCase();
    if (/(fail|error|abort|reject|retract|evict|free|oom|timeout|失败|错误|拒绝|回收|释放)/.test(text)) return "red";
    if (/(draft|verify|accept|eagle|dflash|dspark|speculative|投机|草稿|验证|接受)/.test(text)) return "pink";
    if (/(\b(?:tp|dp|pp|ep|cp|dcp)\b|rank|nccl|ipc|zmq|grpc|transfer|producer|consumer|bootstrap|disaggreg|分布式|通信|传输|解聚)/.test(text)) return "cyan";
    if (/(kv|cache|memory|pool|tensor|hidden|buffer|address|index|radix|token|缓存|内存|张量|地址|索引)/.test(text)) return "green";
    if (/(model|runner|attention|kernel|cuda|forward|worker|layer|graph|device|模型|执行|算子|设备)/.test(text)) return "purple";
    if (/(scheduler|batch|guard|admit|select|policy|schedule|if\b|match|调度|准入|策略|条件)/.test(text)) return "orange";
    if (/(test|benchmark|ci|deploy|source|assert|profile|测试|基准|证据|部署)/.test(text)) return "yellow";
    if (element && element.querySelector && element.querySelector("polygon")) return "orange";
    var fallback = {
      architecture: ["blue", "orange", "purple", "green", "cyan"],
      component: ["blue", "purple", "green", "orange"],
      sequence: ["blue", "orange", "purple", "cyan"],
      state: ["blue", "orange", "green", "red"],
      data: ["blue", "green", "purple", "cyan"],
      code: ["blue", "orange", "green", "purple"]
    };
    var palette = fallback[kind] || fallback.component;
    return palette[index % palette.length];
  }

  function decorateMermaid(svg, kind) {
    Array.from(svg.querySelectorAll("g.node")).forEach(function (node, index) {
      node.dataset.sglTone = mermaidTone(node.textContent, kind, index, node);
    });
    Array.from(svg.querySelectorAll("g.statediagram-state, g.classGroup")).forEach(function (node, index) {
      if (!node.dataset.sglTone) node.dataset.sglTone = mermaidTone(node.textContent, kind, index, node);
    });
    Array.from(svg.querySelectorAll("g.cluster")).forEach(function (cluster, index) {
      cluster.dataset.sglTone = mermaidTone(cluster.textContent, kind, index, cluster);
    });

    var actorTexts = Array.from(svg.querySelectorAll("text.actor"));
    var actorRects = Array.from(svg.querySelectorAll("rect.actor"));
    actorTexts.forEach(function (actor, index) {
      var tone = mermaidTone(actor.textContent, "sequence", index, actor);
      actor.dataset.sglTone = tone;
      if (actorRects[index]) actorRects[index].dataset.sglTone = tone;
    });
    Array.from(svg.querySelectorAll("rect.note")).forEach(function (note) {
      note.dataset.sglTone = "yellow";
    });
  }

  function mermaidViewBox(svg) {
    var box = svg.viewBox && svg.viewBox.baseVal;
    if (box && box.width > 0 && box.height > 0) return { width: box.width, height: box.height };
    var raw = (svg.getAttribute("viewBox") || "").trim().split(/[ ,]+/).map(Number);
    return raw.length === 4 && raw[2] > 0 && raw[3] > 0
      ? { width: raw[2], height: raw[3] }
      : null;
  }

  function normalizeMermaidDiagram(diagram) {
    var frame = diagram.closest("[data-diagram-frame]");
    var viewport = frame && frame.querySelector(".diagram-frame__viewport");
    var svg = diagram.querySelector("svg");
    if (!frame || !viewport || !svg) return;

    var size = mermaidViewBox(svg);
    var status = frame.querySelector("[data-diagram-status]");
    if (!size) {
      frame.dataset.diagramError = "true";
      if (status) status.textContent = "无法读取图形尺寸";
      return;
    }

    decorateMermaid(svg, frame.dataset.diagramKind || "generic");
    var available = Math.max(280, viewport.clientWidth - 40);
    var fitScale = Math.min(1, available / size.width);
    var effectiveFont = MERMAID_BASE_FONT_PX * fitScale;
    var layout = effectiveFont >= MERMAID_MIN_EFFECTIVE_FONT_PX ? "fit" : "scroll";
    var displayWidth = layout === "fit" ? Math.min(size.width, available) : size.width;

    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.maxWidth = "none";
    svg.style.width = Math.ceil(displayWidth) + "px";
    svg.style.height = "auto";
    frame.dataset.diagramLayout = layout;
    frame.dataset.vbWidth = String(Math.round(size.width));
    frame.dataset.vbHeight = String(Math.round(size.height));
    frame.dataset.fitScale = fitScale.toFixed(3);
    frame.dataset.effectiveFontPx = (layout === "fit" ? effectiveFont : MERMAID_BASE_FONT_PX).toFixed(1);
    frame.removeAttribute("data-diagram-error");
    if (status) {
      status.textContent = layout === "fit"
        ? "正文可读 · " + Math.round(effectiveFont * 10) / 10 + "px"
        : "1:1 可读 · 横向滚动";
    }
  }

  function normalizeMermaidDiagrams(diagrams) {
    diagrams.forEach(normalizeMermaidDiagram);
  }

  function initMermaid() {
    var diagrams = Array.from(document.querySelectorAll("pre.mermaid"));
    if (!diagrams.length || !window.mermaid) return;

    var sources = new Map();
    var renderQueue = Promise.resolve();
    var requestedTheme = "";
    var renderSerial = 0;
    diagrams.forEach(function (diagram) {
      sources.set(diagram, diagram.textContent.trim());
    });

    function showRenderError(diagram, error) {
      diagram.textContent = sources.get(diagram);
      diagram.removeAttribute("data-processed");
      diagram.dataset.mermaidError = "true";
      var frame = diagram.closest("[data-diagram-frame]");
      if (frame) {
        frame.dataset.diagramError = "true";
        var status = frame.querySelector("[data-diagram-status]");
        if (status) status.textContent = "图形渲染失败 · 下方保留原始图定义";
      }
      console.error("Mermaid render failed", error);
    }

    function render(resolvedTheme) {
      var normalizedTheme = resolvedTheme === "light" ? "light" : "dark";
      if (requestedTheme === normalizedTheme) return;
      requestedTheme = normalizedTheme;

      renderQueue = renderQueue.then(function () {
        window.mermaid.initialize({
          startOnLoad: false,
          securityLevel: "loose",
          suppressErrorRendering: true,
          maxTextSize: 200000,
          theme: "base",
          themeVariables: mermaidThemeVariables(normalizedTheme),
          flowchart: {
            htmlLabels: true,
            curve: "linear",
            nodeSpacing: 38,
            rankSpacing: 50,
            wrappingWidth: 220,
            useMaxWidth: false
          },
          sequence: {
            useMaxWidth: false,
            wrap: true,
            width: 150,
            mirrorActors: false
          }
        });

        // Render each definition with an observable result. mermaid.run with
        // suppressErrors can empty an invalid block and leave data-processed=true,
        // which falsely looks complete to both the viewer and browser checks.
        return diagrams.reduce(function (queue, diagram) {
          return queue.then(function () {
            return window.mermaid.render("sglang-mermaid-" + (++renderSerial), sources.get(diagram))
              .then(function (result) {
                diagram.innerHTML = result.svg;
                if (!diagram.querySelector("svg")) throw new Error("No SVG returned for diagram");
                if (result.bindFunctions) result.bindFunctions(diagram);
                diagram.dataset.processed = "true";
                delete diagram.dataset.mermaidError;
              }).catch(function (error) {
                showRenderError(diagram, error);
              });
          });
        }, Promise.resolve());
      }).then(function () {
        normalizeMermaidDiagrams(diagrams);
        window.dispatchEvent(new CustomEvent("sglang:mermaid-rendered", {
          detail: { theme: normalizedTheme }
        }));
      }).catch(function (error) {
        diagrams.forEach(function (diagram) { showRenderError(diagram, error); });
      });
    }

    window.addEventListener("sglang:theme", function (event) {
      render(event.detail && event.detail.resolved);
    });
    if ("ResizeObserver" in window) {
      var resizeFrame = 0;
      var observer = new ResizeObserver(function () {
        window.cancelAnimationFrame(resizeFrame);
        resizeFrame = window.requestAnimationFrame(function () {
          normalizeMermaidDiagrams(diagrams);
        });
      });
      diagrams.forEach(function (diagram) {
        var viewport = diagram.closest("[data-diagram-frame]") &&
          diagram.closest("[data-diagram-frame]").querySelector(".diagram-frame__viewport");
        if (viewport) observer.observe(viewport);
      });
    }
    render(document.documentElement.dataset.theme || "dark");
  }

  initTheme();
  initNavigationDrawer();
  initCurrentPage();
  initExternalLinks();
  initMermaid();

  Course.theme = { apply: applyTheme };
})();
