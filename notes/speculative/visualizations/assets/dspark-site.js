(() => {
  "use strict";

  const pages = [
    { id: "00", file: "dspark-complete-execution-flow.html", title: "完整执行地图", parent: null, kind: "main", order: 0 },
    { id: "01", file: "dspark-forward-isolation.html", title: "Scheduler 隔离", parent: null, kind: "main", order: 1 },
    { id: "01A", file: "dspark-overlap-nonoverlap-timeline.html", title: "Overlap / Non-overlap 时间线", parent: "01", kind: "topic", order: 1.1 },
    { id: "02", file: "dspark-prefill-bootstrap.html", title: "Prefill 启动", parent: null, kind: "main", order: 2 },
    { id: "03", file: "dspark-draft-proposal.html", title: "Draft 提案", parent: null, kind: "main", order: 3 },
    { id: "03A", file: "dspark-markov-head-internals.html", title: "Markov Head 内部机制", parent: "03", kind: "topic", order: 3.1 },
    { id: "04", file: "dspark-confidence-budget-layout.html", title: "置信度 / 预算 / 布局", parent: null, kind: "main", order: 4 },
    { id: "04A", file: "dspark-budget-history-sps.html", title: "历史置信度与 SPS 预算实验室", parent: "04", kind: "topic", order: 4.1 },
    { id: "04B", file: "dspark-survival-topk-layout.html", title: "当前 Survival / Top-k / Layout", parent: "04", kind: "topic", order: 4.2 },
    { id: "05", file: "dspark-verify-step-through.html", title: "Target 验证", parent: null, kind: "main", order: 5 },
    { id: "06", file: "dspark-accept-finalize.html", title: "接受 / 收尾", parent: null, kind: "main", order: 6 },
    { id: "06A", file: "dspark-sampling-rejection-lab.html", title: "拒绝采样修正实验室", parent: "06", kind: "topic", order: 6.1 },
    { id: "07", file: "dspark-target-hidden-kv-explainer.html", title: "隐藏状态 / Draft KV / 滚动锚点", parent: null, kind: "main", order: 7 },
    { id: "08", file: "dspark-folded-fast-path.html", title: "折叠快路", parent: null, kind: "main", order: 8 },
    { id: "08A", file: "dspark-folded-capture-replay.html", title: "折叠捕获与回放", parent: "08", kind: "topic", order: 8.1 },
  ].sort((a, b) => a.order - b.order);
  // Each visual has one source-note owner and one family-course handoff.
  const related = {
    "00": ["sglang-draft-target-verify-lifecycle.html", "index.html"],
    "01": ["sglang-draft-target-verify-lifecycle.html", "common-contract.html"],
    "01A": ["sglang-dspark-single-round-decode.html", "dspark-folded-overlap.html"],
    "02": ["sglang-dspark-single-round-decode.html", "dspark-bootstrap-proposal.html"],
    "03": ["sglang-dspark-single-round-decode.html", "dspark-bootstrap-proposal.html"],
    "03A": ["sglang-dspark-single-round-decode.html", "dspark-bootstrap-proposal.html"],
    "04": ["sglang-dspark-single-round-decode.html", "dspark-confidence-layout.html"],
    "04A": ["sglang-dspark-single-round-decode.html", "dspark-confidence-layout.html"],
    "04B": ["sglang-dspark-single-round-decode.html", "dspark-confidence-layout.html"],
    "05": ["sglang-dspark-verify-visual-guide.html", "dspark-verify-accept-kv.html"],
    "06": ["sglang-dspark-single-round-decode.html", "dspark-verify-accept-kv.html"],
    "06A": ["sglang-dspark-single-round-decode.html", "dspark-verify-accept-kv.html"],
    "07": ["sglang-dspark-single-round-decode.html", "dspark-verify-accept-kv.html"],
    "08": ["sglang-dspark-single-round-decode.html", "dspark-folded-overlap.html"],
    "08A": ["sglang-dspark-single-round-decode.html", "dspark-folded-overlap.html"],
  };
  const validRoutes = new Set(["static-greedy", "compact-sampling", "folded-greedy"]);
  const pageId = document.body.dataset.page || "00";
  const params = new URLSearchParams(location.search);
  let route = validRoutes.has(params.get("route")) ? params.get("route") : "compact-sampling";

  const pageIndex = Math.max(0, pages.findIndex((page) => page.id === pageId));
  const currentPage = pages[pageIndex];
  const currentStageId = currentPage.kind === "topic" ? currentPage.parent : currentPage.id;
  const mainPages = pages.filter((page) => page.kind === "main");
  const topicPages = pages.filter((page) => page.kind === "topic");
  const routeLabel = {
    "static-greedy": "STATIC · 贪心",
    "compact-sampling": "COMPACT · 随机采样",
    "folded-greedy": "FOLDED · 贪心",
  };

  function withRoute(file, hash = "") {
    const url = new URL(file, location.href);
    url.searchParams.set("route", route);
    url.hash = hash || url.hash;
    return url.pathname.split("/").pop() + url.search + url.hash;
  }

  function buildNav() {
    const mount = document.querySelector("[data-site-nav]");
    if (!mount) return;
    const prev = pages[pageIndex - 1];
    const next = pages[pageIndex + 1];
    mount.className = "site-nav";
    mount.innerHTML = `
      <div class="site-nav__inner">
        <a class="site-brand" data-nav href="${withRoute(pages[0].file)}">dSpark 实验</a>
        <div class="site-nav__step">
          <a class="site-nav__button" ${prev ? `data-nav href="${withRoute(prev.file)}"` : 'aria-disabled="true"'}>← ${prev ? prev.id : ""}</a>
          <a class="site-nav__button" ${next ? `data-nav href="${withRoute(next.file)}"` : 'aria-disabled="true"'}>${next ? next.id : ""} →</a>
        </div>
        <select class="site-nav__select" aria-label="选择 dSpark 学习页面">
          <optgroup label="主阶段 00–08">
            ${mainPages.map((page) => `<option value="${page.file}" ${page.id === pageId ? "selected" : ""}>${page.id} · ${page.title}</option>`).join("")}
          </optgroup>
          <optgroup label="专题深入页">
            ${topicPages.map((page) => `<option value="${page.file}" ${page.id === pageId ? "selected" : ""}>${page.id} · ${page.title}</option>`).join("")}
          </optgroup>
        </select>
        <span class="site-nav__route">${routeLabel[route]}</span>
        <nav class="site-nav__related" aria-label="跨专题阅读">
          <a href="sglang-course-v2/${related[pageId][1]}">对应家族课程</a>
          <a href="../${related[pageId][0]}">深入源码笔记</a>
        </nav>
        <div class="site-nav__stages" aria-label="dSpark 主阶段">
          ${mainPages.map((page) => `<a data-nav href="${withRoute(page.file)}" ${page.id === currentStageId ? 'aria-current="step"' : ""}>${page.id}</a>`).join("")}
        </div>
      </div>`;

    mount.querySelector("select").addEventListener("change", (event) => {
      location.href = withRoute(event.target.value);
    });
  }

  function buildLegend() {
    document.querySelectorAll("[data-semantic-legend]").forEach((mount) => {
      mount.className = "semantic-legend";
      mount.innerHTML = [
        ["main", "主链 · 调用/提交"], ["data", "数据 · 对象/状态"],
        ["guard", "分支 · 条件/临时视图"], ["fail", "错误 · 拒绝/异常"],
        ["cuda", "设备 · CUDA 图/异步"], ["off", "未启用 · 非当前路径/填充"],
      ].map(([kind, label]) => `<span class="semantic-key" data-kind="${kind}">${label}</span>`).join("");
    });
  }

  function syncRoute({ updateUrl = false } = {}) {
    document.documentElement.dataset.route = route;
    if (updateUrl) {
      const url = new URL(location.href);
      url.searchParams.set("route", route);
      history.replaceState(null, "", url);
    }
    document.querySelectorAll("[data-route-choice]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.routeChoice === route));
    });
    document.querySelectorAll("[data-routes]").forEach((element) => {
      element.dataset.active = String(element.dataset.routes.split(/\s+/).includes(route));
    });
    document.querySelectorAll("a[data-nav]").forEach((anchor) => {
      const source = anchor.getAttribute("href");
      if (!source || source.startsWith("#")) return;
      const url = new URL(source, location.href);
      url.searchParams.set("route", route);
      anchor.href = url.pathname.split("/").pop() + url.search + url.hash;
    });
    const label = document.querySelector(".site-nav__route");
    if (label) label.textContent = routeLabel[route];
    dispatchEvent(new CustomEvent("dspark:route", { detail: { route } }));
  }

  function bindRouteControls() {
    document.addEventListener("click", (event) => {
      const button = event.target.closest("[data-route-choice]");
      if (!button || !validRoutes.has(button.dataset.routeChoice)) return;
      route = button.dataset.routeChoice;
      syncRoute({ updateUrl: true });
    });
  }

  function createDialog() {
    if (!document.querySelector("[data-zoom-target]")) return;
    const dialog = document.createElement("dialog");
    dialog.className = "diagram-dialog";
    dialog.innerHTML = `
      <div class="diagram-dialog__bar">
        <strong>图解 · 放大视图</strong>
        <button type="button" data-dialog-close>关闭 ×</button>
      </div>
      <div class="diagram-dialog__content"></div>`;
    (document.querySelector(".dspark-lab") || document.body).append(dialog);
    let opener = null;
    let cloneSerial = 0;

    function cloneDiagram(target) {
      const clone = target.cloneNode(true);
      const prefix = `zoom-${++cloneSerial}-`;
      const idMap = new Map();
      const originals = [target, ...target.querySelectorAll("[id]")];
      const layoutProps = [
        "position", "inset", "left", "right", "top", "bottom",
        "width", "height", "min-width", "min-height", "max-width", "max-height",
      ];

      originals.forEach((original) => {
        const oldId = original.id;
        const copy = target === original
          ? clone
          : clone.querySelector(`#${CSS.escape(oldId)}`);
        if (!copy) return;
        const computed = getComputedStyle(original);
        layoutProps.forEach((prop) => copy.style.setProperty(prop, computed.getPropertyValue(prop)));
        const newId = `${prefix}${oldId}`;
        idMap.set(oldId, newId);
        copy.id = newId;
      });

      const idrefAttrs = ["aria-labelledby", "aria-describedby", "aria-controls", "aria-owns"];
      const referenceAttrs = [
        "href", "xlink:href", "marker-start", "marker-mid", "marker-end",
        "clip-path", "filter", "fill", "stroke",
      ];
      [clone, ...clone.querySelectorAll("*")].forEach((node) => {
        idrefAttrs.forEach((attr) => {
          const value = node.getAttribute?.(attr);
          if (!value) return;
          const rewritten = value
            .split(/\s+/)
            .map((id) => idMap.get(id) || id)
            .join(" ");
          node.setAttribute(attr, rewritten);
        });
        referenceAttrs.forEach((attr) => {
          const value = node.getAttribute?.(attr);
          if (!value || !value.includes("#")) return;
          let rewritten = value;
          idMap.forEach((nextId, oldId) => {
            rewritten = rewritten.replaceAll(`#${oldId}`, `#${nextId}`);
          });
          node.setAttribute(attr, rewritten);
        });
      });
      return clone;
    }

    document.addEventListener("click", (event) => {
      const button = event.target.closest("[data-zoom-target]");
      if (!button) return;
      const target = document.querySelector(button.dataset.zoomTarget);
      if (!target) return;
      opener = button;
      const clone = cloneDiagram(target);
      dialog.querySelector(".diagram-dialog__content").replaceChildren(clone);
      dialog.showModal();
      dialog.querySelector("[data-dialog-close]").focus();
    });
    dialog.querySelector("[data-dialog-close]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener("close", () => opener?.focus());
  }

  function setupArchitectureStories() {
    document.querySelectorAll("[data-architecture-story]").forEach((story) => {
      const tabs = [...story.querySelectorAll("[data-arch-scene]")];
      const panels = [...story.querySelectorAll("[data-arch-panel]")];
      const progress = story.querySelector("[data-arch-progress]");
      const labels = ["入口与动态接收者", "协调器内部职责", "模型执行与结果闭环"];
      let active = 0;

      function render(next, { focus = false } = {}) {
        active = Math.max(0, Math.min(panels.length - 1, Number(next)));
        tabs.forEach((tab, index) => {
          const selected = index === active;
          tab.setAttribute("aria-selected", String(selected));
          tab.tabIndex = selected ? 0 : -1;
        });
        panels.forEach((panel, index) => panel.setAttribute("aria-hidden", String(index !== active)));
        if (progress) progress.textContent = `${active + 1} / ${panels.length} · ${labels[active]}`;
        story.style.setProperty("--arch-progress", `${(active + 1) / panels.length * 100}%`);
        if (focus) tabs[active]?.focus();
      }

      story.addEventListener("click", (event) => {
        const control = event.target.closest("[data-arch-scene],[data-arch-next]");
        if (!control) return;
        render(control.dataset.archScene ?? control.dataset.archNext, { focus: control.matches("[data-arch-scene]") });
      });
      story.addEventListener("keydown", (event) => {
        if (!event.target.matches("[data-arch-scene]") || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
        event.preventDefault();
        render(active + (event.key === "ArrowRight" ? 1 : -1), { focus: true });
      });
      story.addEventListener("pointerover", (event) => {
        const source = event.target.closest("[data-object-ref]");
        if (!source) return;
        story.querySelectorAll(`[data-object-ref="${CSS.escape(source.dataset.objectRef)}"]`).forEach((node) => node.dataset.linked = "true");
      });
      story.addEventListener("pointerout", (event) => {
        const source = event.target.closest("[data-object-ref]");
        if (!source) return;
        story.querySelectorAll(`[data-object-ref="${CSS.escape(source.dataset.objectRef)}"]`).forEach((node) => delete node.dataset.linked);
      });
      render(0);
    });
  }

  buildNav();
  buildLegend();
  bindRouteControls();
  createDialog();
  setupArchitectureStories();
  syncRoute();

  window.DSparkSite = {
    get route() { return route; },
    pages: pages.map(({ id, file, title, parent, kind, order }) => ({ id, file, title, parent, kind, order })),
    withRoute,
  };
})();
