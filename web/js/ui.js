// Small DOM helpers and the shared player components.

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** h("div.cls", {attrs}, ...children) */
export function h(tag, attrs, ...children) {
  const [head, ...classes] = tag.split(".");
  const [name, id] = head.split("#");
  const el = document.createElement(name || "div");
  if (id) el.id = id;
  if (classes.length) el.className = classes.join(" ");
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className += (el.className ? " " : "") + v;
    else if (k === "style" && typeof v === "object") {
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith("--")) el.style.setProperty(prop, val);   // custom properties need setProperty
        else el.style[prop] = val;
      }
    }
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function money(x) {
  if (x === null || x === undefined || !isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a >= 1e6) return `€${(x / 1e6).toFixed(a >= 1e8 ? 0 : 1)}M`;
  if (a >= 1e3) return `€${Math.round(x / 1e3)}K`;
  return `€${Math.round(x)}`;
}

export const round = (x) => (x === null || x === undefined ? "–" : Math.round(x));
export const signed = (x, d = 1) => (x === null || x === undefined ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);

export function debounce(fn, ms = 250) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

const SILHOUETTE = `<svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="36" r="20" fill="currentColor"/><path d="M12 100c2-24 18-36 38-36s36 12 38 36z" fill="currentColor"/></svg>`;

/** Round face: the player's photo from the dataset, or a drawn stand-in. */
export function face(p, cls = "") {
  const wrap = h(`div.face${cls ? "." + cls : ""}`, { style: { "--c": `var(--${p.tier || "silver"})` } });
  const fallback = h("div.fallback", { html: SILHOUETTE });
  wrap.append(fallback);
  if (p.face) {
    const img = h("img", { src: p.face, alt: "", loading: "lazy", decoding: "async", referrerpolicy: "no-referrer" });
    img.addEventListener("load", () => { fallback.remove(); });
    img.addEventListener("error", () => img.remove());
    wrap.append(img);
  }
  return wrap;
}

/** "EA FC 27" -> "FC 27" where space is tight. */
export const shortName = (name) => String(name).replace(/^EA /, "");

/** Pointer-driven tilt with a floodlight sheen. Returns a detach function. */
export function tilt(el, max = 10) {
  if (reducedMotion()) return () => {};
  const move = (e) => {
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    el.style.setProperty("--ry", `${(px - 0.5) * 2 * max}deg`);
    el.style.setProperty("--rx", `${(0.5 - py) * 2 * max}deg`);
    el.style.setProperty("--mx", `${px * 100}%`);
    el.style.setProperty("--my", `${py * 100}%`);
  };
  const enter = () => el.classList.add("live");
  const leave = () => {
    el.classList.remove("live");
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  };
  el.addEventListener("pointerenter", enter);
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerleave", leave);
  return () => {
    el.removeEventListener("pointerenter", enter);
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerleave", leave);
  };
}

/** The player card used in every grid. meta gives the edition names for the forecasts. */
export function playerCard(p, meta, { extra } = {}) {
  const card = h(`a.pcard.${p.tier}`, { href: `#/player/${p.id}`, "aria-label": `${p.name}, ${p.pos}, rated ${p.overall}` },
    h("div.pcard-top",
      h("div", h("div.pcard-ovr", p.overall), h("div.pcard-pos", p.pos)),
      h("div.pcard-meta", { style: { textAlign: "right" } }, `${round(p.age)} yrs`, h("br"), money(p.value))),
    h("div.pcard-face", face(p)),
    h("div",
      h("div.pcard-name", p.name),
      h("div.pcard-club", p.club || "Free agent")),
    h("div.pcard-fc",
      h("div", h("small", shortName(meta.next_name)), h("b", round(p.pred_t1))),
      h("div", h("small", shortName(meta.next2_name)), h("b", round(p.pred_t2))),
      h("div", h("small", "Potential"), h("b", { style: { color: "var(--chalk-2)" } }, p.potential))),
    extra || null,
  );
  tilt(card);
  return h("div.pcard-wrap", card);
}

/** Full-pitch markings in a 68 x 100 box (own goal at the bottom). */
export function pitchLines() {
  const end = (flip) => {
    const y = (v) => (flip ? 100 - v : v);
    return `<rect x="13.84" y="${Math.min(y(84.3), y(100))}" width="40.32" height="15.7"/>
      <rect x="24.84" y="${Math.min(y(94.8), y(100))}" width="18.32" height="5.2"/>
      <path d="M26.47 ${y(84.3)}A9.15 9.15 0 0 ${flip ? 0 : 1} 41.53 ${y(84.3)}"/>
      <circle cx="34" cy="${y(90)}" r=".5" fill="rgba(238,243,234,.7)" stroke="none"/>`;
  };
  return `<svg class="lines" viewBox="0 0 68 100" preserveAspectRatio="none" aria-hidden="true" fill="none" stroke="rgba(238,243,234,.5)" stroke-width=".35" vector-effect="non-scaling-stroke">
    <path d="M0 50h68"/><circle cx="34" cy="50" r="9.15"/><circle cx="34" cy="50" r=".5" fill="rgba(238,243,234,.7)" stroke="none"/>
    ${end(false)}${end(true)}
  </svg>`;
}

export function notice(err) {
  const msg = err?.message || String(err);
  const code = err?.code === "not_built";
  return h("div.notice", { role: "alert" },
    h("strong", code ? "The data hasn't been built yet." : "Something went wrong."),
    h("p", { style: { margin: "6px 0 0" }, html: escapeHtml(msg).replace(/`([^`]+)`/g, "<code>$1</code>") }));
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function setTitle(t) {
  document.title = t ? `${t} | Career Mode Scout` : "Career Mode Scout";
}
