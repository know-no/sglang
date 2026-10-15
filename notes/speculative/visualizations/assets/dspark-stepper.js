(() => {
  "use strict";

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function parseInitial(max) {
    const match = location.hash.match(/^#step-(\d+)$/);
    return Math.min(max, Math.max(0, match ? Number(match[1]) : 0));
  }

  function setup(root) {
    const count = Number(root.dataset.steps || 1);
    const max = Math.max(0, count - 1);
    let step = parseInitial(max);
    let timer = null;
    const fill = root.querySelector("[data-step-progress]");
    const status = root.querySelector("[data-step-status]");
    const play = root.querySelector("[data-step-action='play']");

    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
      if (play) play.textContent = "播放";
    }

    function render({ writeHash = true } = {}) {
      root.dataset.currentStep = String(step);
      root.querySelectorAll("[data-step-node]").forEach((node) => {
        const at = Number(node.dataset.stepNode);
        node.dataset.stepState = at < step ? "past" : at === step ? "active" : "future";
      });
      root.querySelectorAll("[data-step-copy]").forEach((copy) => {
        copy.dataset.current = String(Number(copy.dataset.stepCopy) === step);
      });
      root.querySelectorAll("[data-step-marker]").forEach((marker) => {
        marker.setAttribute("aria-current", Number(marker.dataset.stepMarker) === step ? "step" : "false");
      });
      if (fill) fill.style.width = `${count <= 1 ? 100 : (step / max) * 100}%`;
      if (status) status.textContent = `第 ${step + 1} / ${count} 步`;
      if (writeHash) history.replaceState(null, "", `${location.pathname}${location.search}#step-${step}`);
      root.dispatchEvent(new CustomEvent("dspark:step", { detail: { step, count } }));
    }

    function setStep(next) {
      step = Math.min(max, Math.max(0, next));
      if (step === max) stop();
      render();
    }

    root.addEventListener("click", (event) => {
      const button = event.target.closest("[data-step-action]");
      if (!button) return;
      const action = button.dataset.stepAction;
      if (action === "prev") { stop(); setStep(step - 1); }
      if (action === "next") { stop(); setStep(step + 1); }
      if (action === "reset") { stop(); setStep(0); }
      if (action === "play") {
        if (timer) { stop(); return; }
        if (reduceMotion) { setStep(Math.min(max, step + 1)); return; }
        if (step === max) step = 0;
        play.textContent = "暂停";
        render();
        timer = setInterval(() => setStep(step + 1), 1400);
      }
    });

    root.querySelectorAll("[data-entity]").forEach((entity) => {
      const toggle = (active) => {
        root.querySelectorAll(`[data-entity="${CSS.escape(entity.dataset.entity)}"]`).forEach((peer) => {
          peer.dataset.linked = String(active);
        });
      };
      entity.addEventListener("pointerenter", () => toggle(true));
      entity.addEventListener("pointerleave", () => toggle(false));
      entity.addEventListener("focus", () => toggle(true));
      entity.addEventListener("blur", () => toggle(false));
    });

    render({ writeHash: false });
  }

  document.querySelectorAll("[data-stepper]").forEach(setup);
})();
