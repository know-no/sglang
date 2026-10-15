(function () {
  "use strict";

  var Course = window.SGLangCourse = window.SGLangCourse || {};
  var dialog = document.querySelector("[data-diagram-dialog]");
  if (!dialog) return;

  var viewport = dialog.querySelector("[data-diagram-viewport]");
  var canvas = dialog.querySelector("[data-diagram-canvas]");
  var titleNode = dialog.querySelector("[data-diagram-title]");
  var zoomNode = dialog.querySelector("[data-diagram-zoom]");
  if (!viewport || !canvas) return;

  var state = { scale: 1, x: 0, y: 0, mode: "readable", pointerId: null, lastX: 0, lastY: 0, width: 1, height: 1 };
  var opener = null;
  var serial = 0;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function updateTransform() {
    canvas.style.transform = "translate(" + state.x + "px," + state.y + "px) scale(" + state.scale + ")";
    if (zoomNode) zoomNode.textContent = Math.round(state.scale * 100) + "%";
  }

  function rewriteCloneIds(target) {
    var clone = target.cloneNode(true);
    var prefix = "diagram-clone-" + (++serial) + "-";
    var idMap = new Map();
    var originalNodes = [target].concat(Array.from(target.querySelectorAll("[id]")));

    originalNodes.forEach(function (original) {
      if (!original.id) return;
      var copy = original === target ? clone : clone.querySelector("#" + CSS.escape(original.id));
      if (!copy) return;
      var nextId = prefix + original.id;
      idMap.set(original.id, nextId);
      copy.id = nextId;
    });

    var idRefAttributes = ["aria-labelledby", "aria-describedby", "aria-controls", "aria-owns"];
    var urlRefAttributes = [
      "href", "xlink:href", "marker-start", "marker-mid", "marker-end",
      "clip-path", "filter", "fill", "stroke"
    ];
    [clone].concat(Array.from(clone.querySelectorAll("*"))).forEach(function (node) {
      idRefAttributes.forEach(function (attribute) {
        var value = node.getAttribute && node.getAttribute(attribute);
        if (!value) return;
        node.setAttribute(attribute, value.split(/\s+/).map(function (id) {
          return idMap.get(id) || id;
        }).join(" "));
      });
      urlRefAttributes.forEach(function (attribute) {
        var value = node.getAttribute && node.getAttribute(attribute);
        if (!value || value.indexOf("#") === -1) return;
        var rewritten = value;
        idMap.forEach(function (nextId, oldId) {
          rewritten = rewritten.split("#" + oldId).join("#" + nextId);
        });
        node.setAttribute(attribute, rewritten);
      });
    });

    clone.querySelectorAll("[data-diagram-open]").forEach(function (button) { button.remove(); });
    return clone;
  }

  function svgNaturalSize(svg) {
    var box = svg && svg.viewBox && svg.viewBox.baseVal;
    if (box && box.width > 0 && box.height > 0) return { width: box.width, height: box.height };
    var raw = ((svg && svg.getAttribute("viewBox")) || "").trim().split(/[ ,]+/).map(Number);
    return raw.length === 4 && raw[2] > 0 && raw[3] > 0
      ? { width: raw[2], height: raw[3] }
      : { width: 1, height: 1 };
  }

  function place(scale, mode) {
    state.scale = clamp(scale, 0.12, 6);
    var renderedWidth = state.width * state.scale;
    var renderedHeight = state.height * state.scale;
    state.x = renderedWidth < viewport.clientWidth
      ? (viewport.clientWidth - renderedWidth) / 2
      : 24;
    state.y = renderedHeight < viewport.clientHeight
      ? (viewport.clientHeight - renderedHeight) / 2
      : 24;
    state.mode = mode;
    updateTransform();
  }

  function fit() {
    if (!canvas.firstElementChild) return;
    var inset = 36;
    place(Math.min(
      (viewport.clientWidth - inset * 2) / state.width,
      (viewport.clientHeight - inset * 2) / state.height,
      1.6
    ), "fit");
  }

  function readable() {
    if (!canvas.firstElementChild) return;
    var inset = 36;
    var fitScale = Math.min(
      (viewport.clientWidth - inset * 2) / state.width,
      (viewport.clientHeight - inset * 2) / state.height,
      1.6
    );
    /* Keep Mermaid's 14px base labels at >=12px.  Wide diagrams open as a
       pannable readable canvas instead of a microscopic whole-graph preview. */
    place(Math.max(fitScale, 12 / 14), "readable");
  }

  function reset() {
    place(1, "manual");
  }

  function zoomAt(nextScale, clientX, clientY) {
    var rect = viewport.getBoundingClientRect();
    var next = clamp(nextScale, 0.12, 6);
    var pointX = typeof clientX === "number" ? clientX - rect.left : viewport.clientWidth / 2;
    var pointY = typeof clientY === "number" ? clientY - rect.top : viewport.clientHeight / 2;
    var worldX = (pointX - state.x) / state.scale;
    var worldY = (pointY - state.y) / state.scale;
    state.x = pointX - worldX * next;
    state.y = pointY - worldY * next;
    state.scale = next;
    state.mode = "manual";
    updateTransform();
  }

  function open(target, label) {
    if (!target) return;
    var originalSvg = target.matches && target.matches("svg") ? target : target.querySelector("svg");
    if (!originalSvg) return;
    opener = document.activeElement;
    var clone = rewriteCloneIds(originalSvg);
    var size = svgNaturalSize(originalSvg);
    state.width = size.width;
    state.height = size.height;
    clone.removeAttribute("width");
    clone.removeAttribute("height");
    clone.style.maxWidth = "none";
    clone.style.width = Math.ceil(size.width) + "px";
    clone.style.height = Math.ceil(size.height) + "px";
    canvas.replaceChildren(clone);
    if (titleNode) titleNode.textContent = label || originalSvg.getAttribute("aria-label") || "图解放大视图";
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    var closeButton = dialog.querySelector("[data-diagram-action=\"close\"]");
    if (closeButton) closeButton.focus();
    window.requestAnimationFrame(readable);
  }

  function close() {
    if (typeof dialog.close === "function") dialog.close();
    else {
      dialog.removeAttribute("open");
      canvas.replaceChildren();
      if (opener && opener.focus) opener.focus();
    }
  }

  function enhanceFigures() {
    document.querySelectorAll("[data-diagram-viewable], [data-diagram-frame]").forEach(function (figure, index) {
      var target = figure.querySelector("[data-diagram-content]") ||
        figure.querySelector(".diagram-frame__viewport") ||
        figure.querySelector(".mermaid");
      var toolbar = figure.querySelector(".diagram-frame__toolbar, .diagram-toolbar");
      if (!target || !toolbar) return;
      if (!target.id) target.id = "diagram-content-" + (index + 1);
      var button = toolbar.querySelector("[data-diagram-open]");
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.dataset.diagramOpen = "";
        button.textContent = "放大 / 平移";
        toolbar.appendChild(button);
      }
      button.classList.add("button", "button--quiet");
      button.dataset.diagramTarget = target.id;
      button.setAttribute("aria-haspopup", "dialog");
      if (dialog.id) button.setAttribute("aria-controls", dialog.id);
    });
  }

  document.addEventListener("click", function (event) {
    var openButton = event.target.closest("[data-diagram-open]");
    if (openButton) {
      var targetId = openButton.dataset.diagramTarget || "";
      var figure = openButton.closest("[data-diagram-viewable], [data-diagram-frame], figure");
      var target = targetId ? document.getElementById(targetId) : figure && (
        figure.querySelector("[data-diagram-content]") ||
        figure.querySelector(".diagram-frame__viewport") ||
        figure.querySelector(".mermaid") ||
        figure
      );
      var caption = figure && figure.querySelector("figcaption");
      open(target, openButton.dataset.diagramLabel || (caption && caption.textContent.trim()));
      return;
    }

    var inlineSvg = event.target.closest && event.target.closest(".mermaid svg");
    if (inlineSvg) {
      var inlineFigure = inlineSvg.closest("[data-diagram-frame], figure");
      var inlineCaption = inlineFigure && inlineFigure.querySelector("figcaption");
      open(inlineSvg, inlineCaption && inlineCaption.textContent.trim());
      return;
    }

    var actionButton = event.target.closest("[data-diagram-action]");
    if (!actionButton || !dialog.contains(actionButton)) return;
    var action = actionButton.dataset.diagramAction;
    if (action === "close") close();
    else if (action === "zoom-in") zoomAt(state.scale * 1.2);
    else if (action === "zoom-out") zoomAt(state.scale / 1.2);
    else if (action === "reset") reset();
    else if (action === "fit") fit();
  });

  dialog.addEventListener("click", function (event) { if (event.target === dialog) close(); });
  dialog.addEventListener("close", function () {
    canvas.replaceChildren();
    if (opener && opener.focus) opener.focus();
  });

  viewport.addEventListener("wheel", function (event) {
    event.preventDefault();
    zoomAt(state.scale * (event.deltaY < 0 ? 1.12 : 0.89), event.clientX, event.clientY);
  }, { passive: false });

  viewport.addEventListener("pointerdown", function (event) {
    if (event.button !== 0) return;
    state.pointerId = event.pointerId;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    viewport.dataset.dragging = "true";
    viewport.setPointerCapture(event.pointerId);
  });

  viewport.addEventListener("pointermove", function (event) {
    if (state.pointerId !== event.pointerId) return;
    state.x += event.clientX - state.lastX;
    state.y += event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    state.mode = "manual";
    updateTransform();
  });

  function releasePointer(event) {
    if (state.pointerId !== event.pointerId) return;
    state.pointerId = null;
    viewport.dataset.dragging = "false";
    if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
  }
  viewport.addEventListener("pointerup", releasePointer);
  viewport.addEventListener("pointercancel", releasePointer);

  dialog.addEventListener("keydown", function (event) {
    if (event.key === "+" || event.key === "=") { event.preventDefault(); zoomAt(state.scale * 1.2); }
    else if (event.key === "-") { event.preventDefault(); zoomAt(state.scale / 1.2); }
    else if (event.key === "0") { event.preventDefault(); fit(); }
  });

  if ("ResizeObserver" in window) {
    new ResizeObserver(function () {
      if (!dialog.open) return;
      if (state.mode === "fit") fit();
      else if (state.mode === "readable") readable();
    }).observe(viewport);
  }

  enhanceFigures();
  window.addEventListener("sglang:mermaid-rendered", enhanceFigures);
  Course.diagramViewer = { open: open, close: close, fit: fit, readable: readable };
})();
