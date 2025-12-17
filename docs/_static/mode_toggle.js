/**
 * Mode Toggle - Switches between Pipeline Mode and Dynamic Mode documentation pages.
 *
 * Features:
 * - Loads a manifest of page pairs (generated at build time)
 * - Injects a toggle UI at the top of pages with variants
 * - Preserves scroll position when switching modes
 * - Highlights the corresponding sidebar entry for dynamic pages
 */

(function () {
    "use strict";

    const SCROLL_KEY = "mode-toggle-scroll";
    let variantsCache = null;

    // --- URL helpers ---

    function getBaseUrl() {
        const script = document.querySelector('script[src*="_static"]');
        if (script) {
            const match = script.src.match(/^(.*\/_static\/)/);
            if (match) return match[1];
        }
        return "/_static/";
    }

    function normalizePath(path) {
        path = path.replace(/^\//, "").replace(/\/index\.html$/, "/");
        const match = path.match(/(examples\/.*\.html)/);
        return match ? match[1] : path;
    }

    function manifestPathToUrl(manifestPath) {
        const currentPath = window.location.pathname;
        const idx = currentPath.indexOf("/examples/");
        if (idx !== -1) {
            return currentPath.substring(0, idx + 1) + manifestPath;
        }
        return "/" + manifestPath;
    }

    function isDynamicMode(path) {
        return path.includes("_dynamic");
    }

    // --- Manifest loading ---

    async function getVariants() {
        if (variantsCache) return variantsCache;

        try {
            const response = await fetch(getBaseUrl() + "mode_variants.json");
            variantsCache = response.ok ? await response.json() : { variants: {} };
        } catch {
            variantsCache = { variants: {} };
        }
        return variantsCache;
    }

    // --- Scroll position ---

    function saveScrollPosition() {
        sessionStorage.setItem(SCROLL_KEY, window.scrollY.toString());
    }

    function restoreScrollPosition() {
        const saved = sessionStorage.getItem(SCROLL_KEY);
        if (saved !== null) {
            sessionStorage.removeItem(SCROLL_KEY);
            window.scrollTo(0, parseInt(saved, 10));
        }
    }

    // --- Toggle UI ---

    function createToggle(currentMode, variantUrl) {
        const container = document.createElement("div");
        container.className = "mode-toggle-container";

        const nav = document.createElement("nav");
        nav.className = "mode-toggle";
        nav.setAttribute("aria-label", "Mode selection");

        const modes = [
            { name: "Pipeline Mode", mode: "pipeline" },
            { name: "Dynamic Mode", mode: "dynamic" },
        ];

        for (const { name, mode } of modes) {
            const link = document.createElement("a");
            link.textContent = name;
            link.href = currentMode === mode ? "#" : variantUrl;

            if (currentMode === mode) {
                link.className = "active";
                link.setAttribute("aria-current", "page");
            } else {
                link.addEventListener("click", saveScrollPosition);
            }
            nav.appendChild(link);
        }

        container.appendChild(nav);
        return container;
    }

    function injectToggle(currentMode, variantUrl) {
        const article = document.querySelector("article.bd-article");
        if (article) {
            article.insertBefore(createToggle(currentMode, variantUrl), article.firstChild);
        }
    }

    // --- Sidebar highlighting ---

    function highlightSidebarEntry(currentPath, variantUrl) {
        if (!isDynamicMode(currentPath)) return;

        const sidebar = document.querySelector(".bd-sidebar-primary");
        if (!sidebar) return;

        const targetFilename = variantUrl.split("/").pop();

        for (const link of sidebar.querySelectorAll("a.reference.internal")) {
            const href = link.getAttribute("href");
            if (href && href.split("/").pop() === targetFilename) {
                // Mark link and its li as current
                link.classList.add("current", "active");
                const li = link.closest("li");
                if (li) li.classList.add("current", "active");

                // Propagate up the ancestor chain
                for (let el = li?.parentElement; el && el !== sidebar; el = el.parentElement) {
                    if (el.tagName === "UL" || el.tagName === "LI") {
                        el.classList.add("current");
                    } else if (el.tagName === "DETAILS") {
                        el.setAttribute("open", "open");
                    }
                }
                break;
            }
        }
    }

    // --- Init ---

    async function init() {
        restoreScrollPosition();

        const manifest = await getVariants();
        const currentPath = normalizePath(window.location.pathname);
        const variantManifestPath = manifest.variants[currentPath];

        if (variantManifestPath) {
            const currentMode = isDynamicMode(currentPath) ? "dynamic" : "pipeline";
            const variantUrl = manifestPathToUrl(variantManifestPath);

            injectToggle(currentMode, variantUrl);
            highlightSidebarEntry(currentPath, variantUrl);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
