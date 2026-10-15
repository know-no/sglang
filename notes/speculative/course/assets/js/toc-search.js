(function () {
  "use strict";

  var Course = window.SGLangCourse = window.SGLangCourse || {};

  function slugify(text, fallback) {
    var value = String(text || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9\u3400-\u9fff]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return value || fallback;
  }

  function uniqueId(base) {
    var id = base;
    var suffix = 2;
    while (document.getElementById(id)) {
      id = base + "-" + suffix;
      suffix += 1;
    }
    return id;
  }

  function initToc() {
    var mount = document.querySelector("[data-page-toc-list]");
    var source = document.querySelector("[data-toc-source]") || document.querySelector("main");
    if (!mount || !source) return;

    var headings = Array.from(source.querySelectorAll("h2, h3")).filter(function (heading) {
      return !heading.closest("[data-toc-ignore]");
    });
    headings.forEach(function (heading, index) {
      if (!heading.id) heading.id = uniqueId(slugify(heading.textContent, "section-" + (index + 1)));
    });

    if (!mount.querySelector("a")) {
      headings.forEach(function (heading) {
        var item = document.createElement("li");
        var link = document.createElement("a");
        link.href = "#" + heading.id;
        link.dataset.level = heading.tagName === "H3" ? "3" : "2";
        link.textContent = heading.textContent.replace(/\s*#\s*$/, "").trim();
        item.appendChild(link);
        mount.appendChild(item);
      });
    }

    var links = Array.from(mount.querySelectorAll("a[href^=\"#\"]"));
    if (!links.length) return;
    var byId = new Map(links.map(function (link) {
      return [decodeURIComponent(link.getAttribute("href").slice(1)), link];
    }));

    function activate(id) {
      links.forEach(function (link) {
        if (link === byId.get(id)) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }

    if ("IntersectionObserver" in window) {
      var visible = new Map();
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) visible.set(entry.target.id, entry.boundingClientRect.top);
          else visible.delete(entry.target.id);
        });
        var active = Array.from(visible.entries()).sort(function (a, b) { return a[1] - b[1]; })[0];
        if (active) activate(active[0]);
      }, { rootMargin: "-18% 0px -68% 0px", threshold: [0, 1] });
      headings.forEach(function (heading) { observer.observe(heading); });
    }

    if (location.hash) activate(decodeURIComponent(location.hash.slice(1)));
  }

  function normalizeEntry(entry, rootRelative) {
    if (!entry) return null;
    var title = String(entry.title || entry.text || "").trim();
    var url = String(entry.url || entry.href || "").trim();
    if (!title || !url) return null;
    return {
      title: title,
      url: url,
      section: String(entry.section || entry.group || entry.path || ""),
      summary: String(entry.summary || entry.description || ""),
      keywords: [
        String(entry.id || ""),
        url.split("/").pop().replace(/\.html(?:[?#].*)?$/, ""),
        Array.isArray(entry.keywords) ? entry.keywords.join(" ") : String(entry.keywords || ""),
        Array.isArray(entry.headings) ? entry.headings.join(" ") : String(entry.headings || "")
      ].filter(Boolean).join(" "),
      body: String(entry.body || entry.content || ""),
      rootRelative: Boolean(rootRelative)
    };
  }

  function collectSearchIndex() {
    var supplied = window.SGLANG_COURSE_SEARCH_INDEX ||
      window.__SGLANG_COURSE_SEARCH_INDEX__ ||
      window.__SGL_SEARCH_INDEX__ || [];
    var entries = Array.isArray(supplied) ? supplied.map(function (entry) {
      return normalizeEntry(entry, true);
    }).filter(Boolean) : [];

    if (!entries.length) {
      document.querySelectorAll("[data-course-link][href]").forEach(function (link) {
        var entry = normalizeEntry({
          title: link.dataset.searchTitle || link.textContent,
          url: link.getAttribute("href"),
          section: link.dataset.searchSection || "课程目录",
          summary: link.dataset.searchSummary || "",
          keywords: link.dataset.searchKeywords || ""
        }, false);
        if (entry) entries.push(entry);
      });
    }

    var seen = new Set();
    return entries.filter(function (entry) {
      var key = entry.url + "\u0000" + entry.title;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function scoreEntry(entry, query) {
    var q = query.toLocaleLowerCase();
    var title = entry.title.toLocaleLowerCase();
    var section = entry.section.toLocaleLowerCase();
    var summary = entry.summary.toLocaleLowerCase();
    var keywords = entry.keywords.toLocaleLowerCase();
    var body = entry.body.toLocaleLowerCase();
    var terms = q.split(/\s+/).filter(Boolean);
    if (!terms.length) return 0;

    var haystack = [title, section, summary, keywords, body].join(" ");
    if (!terms.every(function (term) { return haystack.indexOf(term) !== -1; })) return -1;

    var score = 0;
    if (title === q) score += 100;
    if (title.indexOf(q) === 0) score += 45;
    if (title.indexOf(q) !== -1) score += 30;
    if (keywords.indexOf(q) !== -1) score += 18;
    if (section.indexOf(q) !== -1) score += 12;
    if (summary.indexOf(q) !== -1) score += 7;
    terms.forEach(function (term) {
      if (title.indexOf(term) !== -1) score += 8;
      if (keywords.indexOf(term) !== -1) score += 4;
    });
    return score;
  }

  function initSearch() {
    var dialog = document.querySelector("[data-search-dialog]");
    if (!dialog) return;
    var form = dialog.querySelector("[data-search-form]");
    var input = dialog.querySelector("[data-search-input]");
    var results = dialog.querySelector("[data-search-results]");
    var status = dialog.querySelector("[data-search-status]");
    var opener = null;
    var entries = collectSearchIndex();

    document.querySelectorAll("[data-search-open]").forEach(function (button) {
      button.setAttribute("aria-haspopup", "dialog");
      if (dialog.id) button.setAttribute("aria-controls", dialog.id);
    });

    function resolveEntryUrl(entry) {
      if (!entry.rootRelative) return entry.url;
      try {
        var rootDocument = new URL(document.body.dataset.rootUrl || "index.html", window.location.href);
        return new URL(entry.url, rootDocument).href;
      } catch (_error) {
        return entry.url;
      }
    }

    function openDialog() {
      opener = document.activeElement;
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
      window.setTimeout(function () {
        if (input) {
          input.focus();
          input.select();
        }
      }, 0);
    }

    function closeDialog() {
      if (typeof dialog.close === "function") dialog.close();
      else dialog.removeAttribute("open");
    }

    function appendResult(entry) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      var path = document.createElement("span");
      var title = document.createElement("span");
      var summary = document.createElement("span");
      link.className = "search-result";
      link.href = resolveEntryUrl(entry);
      path.className = "search-result__path";
      title.className = "search-result__title";
      summary.className = "search-result__summary";
      path.textContent = entry.section;
      title.textContent = entry.title;
      summary.textContent = entry.summary;
      link.append(path, title);
      if (entry.summary) link.append(summary);
      item.appendChild(link);
      results.appendChild(item);
    }

    function render(query) {
      var normalized = String(query || "").trim();
      results.replaceChildren();
      var ranked;
      if (!normalized) {
        ranked = entries.slice(0, 12).map(function (entry) { return { entry: entry, score: 0 }; });
      } else {
        ranked = entries
          .map(function (entry) { return { entry: entry, score: scoreEntry(entry, normalized) }; })
          .filter(function (result) { return result.score >= 0; })
          .sort(function (a, b) { return b.score - a.score || a.entry.title.localeCompare(b.entry.title); })
          .slice(0, 30);
      }
      ranked.forEach(function (result) { appendResult(result.entry); });
      status.textContent = normalized
        ? "找到 " + ranked.length + " 个结果"
        : "输入对象、方法、状态或章节名称；当前显示课程入口";
    }

    document.addEventListener("click", function (event) {
      if (event.target.closest("[data-search-open]")) {
        openDialog();
        render(input ? input.value : "");
      } else if (event.target.closest("[data-search-close]")) {
        closeDialog();
      }
    });

    if (form) form.addEventListener("submit", function (event) { event.preventDefault(); render(input.value); });
    if (input) input.addEventListener("input", function () { render(input.value); });
    dialog.addEventListener("click", function (event) { if (event.target === dialog) closeDialog(); });
    dialog.addEventListener("close", function () { if (opener && opener.focus) opener.focus(); });

    document.addEventListener("keydown", function (event) {
      var target = event.target;
      var isTyping = target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName);
      if ((event.key === "/" && !isTyping) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k")) {
        event.preventDefault();
        openDialog();
        render(input ? input.value : "");
      }
    });

    Course.search = { open: openDialog, entries: entries.slice() };
  }

  initToc();
  initSearch();
})();
