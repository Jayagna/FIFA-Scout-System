// Hash router: #/<page>/<arg>?<query>. Each page module exports render(view, ctx)
// and may return a cleanup function that runs when the page is left.

import { get } from "./api.js";
import { h, notice, reducedMotion } from "./ui.js";

const PAGES = {
  "": () => import("./pages/home.js"),
  wonderkids: () => import("./pages/wonderkids.js"),
  squad: () => import("./pages/squad.js"),
  replacements: () => import("./pages/replacements.js"),
  player: () => import("./pages/player.js"),
};

const view = document.getElementById("view");
let cleanup = null;
let meta = null;
let renderId = 0;

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const [page = "", arg] = path.split("/");
  return { page, arg: arg ? decodeURIComponent(arg) : undefined, query: new URLSearchParams(query) };
}

/** Replace the query part of the current page's hash without re-rendering. */
function setQuery(params) {
  const { page, arg } = parseHash();
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || (Array.isArray(v) && !v.length)) continue;
    q.set(k, Array.isArray(v) ? v.join("|") : v);
  }
  const hash = `#/${page}${arg ? "/" + encodeURIComponent(arg) : ""}${q.toString() ? "?" + q : ""}`;
  history.replaceState(null, "", hash);
}

function markNav(page) {
  for (const a of document.querySelectorAll(".nav a")) {
    const target = a.getAttribute("href").replace(/^#\//, "");
    if (target === page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
}

async function route() {
  const id = ++renderId;
  const { page, arg, query } = parseHash();
  const load = PAGES[page] || PAGES[""];
  if (cleanup) { try { cleanup(); } catch { /* page already gone */ } cleanup = null; }
  markNav(page);
  document.getElementById("top").classList.remove("open");
  view.replaceChildren(h("div.wrap.loading", "Loading…"));
  try {
    if (!meta) meta = await get("meta");
    chip(meta);
    const mod = await load();
    if (id !== renderId) return;
    view.replaceChildren();
    if (!reducedMotion()) view.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "ease-out" });
    cleanup = (await mod.render(view, { meta, arg, query, setQuery, isCurrent: () => id === renderId })) || null;
    if (page) view.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  } catch (err) {
    if (id !== renderId) return;
    view.replaceChildren(h("div.wrap.page", notice(err)));
    console.error(err);
  }
}

function chip(m) {
  const el = document.getElementById("db-chip");
  el.replaceChildren(h("b", m.edition_name), ` database, ${m.players.toLocaleString()} players`);
}

document.querySelector(".menu-btn").addEventListener("click", (e) => {
  const top = document.getElementById("top");
  const open = top.classList.toggle("open");
  e.currentTarget.setAttribute("aria-expanded", String(open));
});
window.addEventListener("hashchange", route);
route();
