/* Progressive enhancement for the pre-rendered, offline Markdown reading pages. */
(() => {
  const root = document.documentElement;
  const isDark = () => root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  const figures = Array.from(document.querySelectorAll('.note-diagram'));
  let renderQueue = Promise.resolve();

  const renderDiagrams = async () => {
    if (!figures.length || !window.mermaid) return;
    mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: isDark() ? 'dark' : 'default' });
    for (const [index, figure] of figures.entries()) {
      const canvas = figure.querySelector('.mermaid');
      const button = figure.querySelector('[data-diagram-open]');
      try {
        const result = await mermaid.render(`note-diagram-${index}`, canvas.dataset.source);
        canvas.innerHTML = result.svg;
        result.bindFunctions?.(canvas);
        canvas.dataset.rendered = 'true';
        delete canvas.dataset.error;
        figure.querySelector('.diagram-error')?.remove();
        button.disabled = false;
      } catch (error) {
        canvas.textContent = canvas.dataset.source;
        canvas.dataset.error = String(error);
        delete canvas.dataset.rendered;
        button.disabled = true;
        if (!figure.querySelector('.diagram-error')) {
          const message = document.createElement('p');
          message.className = 'diagram-error';
          message.textContent = '此图渲染失败，下面保留了图定义。';
          canvas.before(message);
        }
      }
    }
  };
  // The shared course shell owns theme and persistence across every reading surface.
  window.addEventListener('sglang:theme', () => {
    renderQueue = renderQueue.then(renderDiagrams);
  });

  document.querySelectorAll('.note-math').forEach(element => {
    if (!window.katex) return;
    try {
      katex.render(element.dataset.tex, element, {
        displayMode: element.dataset.display === 'true', throwOnError: true, trust: false,
      });
      element.dataset.rendered = 'true';
    } catch (error) {
      element.dataset.error = String(error);
      element.title = '公式渲染失败，显示原始表达式';
    }
  });

  renderQueue = renderQueue.then(renderDiagrams).then(async () => {
    await document.fonts.ready;
    root.dataset.readingReady = 'true';
    // Initial fragment scrolling must account for diagrams and formulas gaining height.
    if (location.hash) {
      try { document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView(); } catch (_) {}
    }
  });
})();
