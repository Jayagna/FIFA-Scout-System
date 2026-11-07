// SVG line charts with a crosshair tooltip. Series colours were checked with the
// dataviz palette validator against the panel surface (#103524), all pairs:
// ochre (model / forecast), blue (EA potential / overall), magenta (no change / potential).

import { h } from "./ui.js";

export const SERIES = { ochre: "#c98500", blue: "#3987e5", magenta: "#d55181" };
const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) el.setAttribute(k, v);
  for (const k of kids) if (k) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
};

/**
 * lineChart({ labels, series: [{ name, color, values, dash }], band: { lo, hi, color },
 *             yFormat, height, ariaLabel })
 * values may contain null (gaps). Returns an element that redraws on resize.
 */
export function lineChart(opts) {
  const box = h("div", { style: { position: "relative" } });
  const tip = h("div", { role: "status", style: {
    position: "absolute", pointerEvents: "none", opacity: 0, transition: "opacity .12s",
    background: "#0a2619", border: "1px solid rgba(238,243,234,.28)", borderRadius: "6px",
    padding: "8px 10px", fontSize: ".8rem", lineHeight: 1.45, minWidth: "140px", zIndex: 3,
  } });
  const legend = h("div.legend", opts.series.map((se) =>
    h("span", h("i", { style: { background: se.color, ...(se.dash ? { backgroundImage: `repeating-linear-gradient(90deg, ${se.color} 0 5px, transparent 5px 8px)`, background: "none" } : {}) } }), se.name)),
    opts.band ? h("span", h("i", { style: { background: opts.band.color, opacity: 0.35, height: "8px" } }), opts.band.name) : null);
  const draw = () => {
    const W = Math.max(280, box.clientWidth), H = opts.height || 260;
    const m = { t: 14, r: opts.endLabels === false ? 14 : 112, b: 34, l: 38 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const n = opts.labels.length;
    const all = opts.series.flatMap((se) => se.values).concat(opts.band ? [...opts.band.lo, ...opts.band.hi] : [])
      .filter((v) => v !== null && v !== undefined && isFinite(v));
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = Math.max(0.5, (hi - lo) * 0.12);
    lo = opts.yMin ?? Math.floor(lo - pad); hi = opts.yMax ?? Math.ceil(hi + pad);
    const x = (i) => m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
    const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.ariaLabel || "" });
    // grid + y axis (recessive)
    const ticks = niceTicks(lo, hi, 4);
    for (const tv of ticks) {
      svg.append(s("line", { x1: m.l, x2: m.l + iw, y1: y(tv), y2: y(tv), stroke: "rgba(238,243,234,.09)" }));
      svg.append(s("text", { x: m.l - 8, y: y(tv) + 4, "text-anchor": "end", fill: "#7f9a8a", "font-size": 11 }, (opts.yFormat || String)(tv)));
    }
    // x labels: thin out when crowded, always keep the last, never let two collide
    const every = Math.ceil(n / Math.max(2, Math.floor(iw / 70)));
    opts.labels.forEach((lab, i) => {
      const lastShown = i === n - 1;
      if (!lastShown && (i % every || (n - 1 - i < every && (n - 1) % every))) return;
      svg.append(s("text", { x: x(i), y: H - 10, "text-anchor": "middle", fill: "#7f9a8a", "font-size": 11 }, lab));
    });
    // band
    if (opts.band) {
      const idx = opts.band.lo.map((v, i) => (v === null || v === undefined ? null : i)).filter((i) => i !== null);
      if (idx.length) {
        const top = idx.map((i) => `${x(i)},${y(opts.band.hi[i])}`);
        const bot = idx.slice().reverse().map((i) => `${x(i)},${y(opts.band.lo[i])}`);
        svg.append(s("polygon", { points: top.concat(bot).join(" "), fill: opts.band.color, opacity: 0.22 }));
      }
    }
    // lines, markers, end labels
    const ends = [];
    for (const se of opts.series) {
      let d = "", pen = false;
      se.values.forEach((v, i) => {
        if (v === null || v === undefined) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(i)},${y(v)}`; pen = true;
      });
      svg.append(s("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": se.dash ? "5 4" : null }));
      se.values.forEach((v, i) => {
        if (v === null || v === undefined) return;
        svg.append(s("circle", { cx: x(i), cy: y(v), r: 4, fill: se.color, stroke: "#103524", "stroke-width": 2 }));
      });
      const last = se.values.map((v, i) => [v, i]).filter(([v]) => v !== null && v !== undefined).pop();
      if (last && opts.endLabels !== false) ends.push({ y: y(last[0]), x: x(last[1]), text: `${se.short || se.name} ${(opts.yFormat || String)(last[0])}`, color: se.color });
    }
    // direct labels: nudge apart so they never collide
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
    for (const e of ends) {
      svg.append(s("rect", { x: e.x + 9, y: e.y - 4, width: 8, height: 2, fill: e.color }));
      svg.append(s("text", { x: e.x + 21, y: e.y + 4, fill: "#eef3ea", "font-size": 11.5, "font-weight": 600 }, e.text));
    }
    // hover layer: crosshair + tooltip, hit areas wider than the marks
    const cross = s("line", { y1: m.t, y2: m.t + ih, stroke: "rgba(238,243,234,.35)", "stroke-dasharray": "3 3", opacity: 0 });
    svg.append(cross);
    for (let i = 0; i < n; i++) {
      const left = i === 0 ? m.l - 10 : (x(i - 1) + x(i)) / 2, right = i === n - 1 ? m.l + iw + 10 : (x(i) + x(i + 1)) / 2;
      const hit = s("rect", { x: left, y: m.t, width: right - left, height: ih, fill: "transparent" });
      const show = () => {
        cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
        tip.replaceChildren(h("strong", opts.labels[i]), ...opts.series.map((se) => se.values[i] === null || se.values[i] === undefined ? null :
          h("div", h("i", { style: { display: "inline-block", width: "10px", height: "3px", background: se.color, marginRight: "6px", verticalAlign: "middle" } }), `${se.name}: `, h("b", (opts.yFormat || String)(se.values[i])))),
          opts.band && opts.band.lo[i] !== null && opts.band.lo[i] !== undefined ? h("div", { style: { color: "#a9c0b2" } }, `${opts.band.name}: ${(opts.yFormat || String)(opts.band.lo[i])}–${(opts.yFormat || String)(opts.band.hi[i])}`) : null);
        tip.style.opacity = 1;
        const tx = x(i) + 12 + 160 > W ? x(i) - 172 : x(i) + 12;
        tip.style.left = `${tx}px`; tip.style.top = `${m.t}px`;
      };
      hit.addEventListener("pointerenter", show);
      hit.addEventListener("pointerleave", () => { tip.style.opacity = 0; cross.setAttribute("opacity", 0); });
      svg.append(hit);
    }
    box.replaceChildren(svg, tip);
  };
  const ro = new ResizeObserver(() => draw());
  ro.observe(box);
  requestAnimationFrame(draw);
  return h("div", box, legend);
}

function niceTicks(lo, hi, count) {
  const span = hi - lo, raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((st) => span / st <= count + 1) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

/** Rolling-origin error chart for the home page. */
export function rollingChart(windows) {
  const labels = windows.map((w) => `${num(w.test)}→${num(w.target)}`);
  const vals = windows.flatMap((w) => [w.model, w.ea_potential, w.persistence]);
  return lineChart({
    labels,
    yMin: Math.floor(Math.min(...vals) * 4) / 4 - 0.25, yMax: Math.ceil(Math.max(...vals) * 4) / 4 + 0.25,
    ariaLabel: "Mean absolute error of each forecasting method for every test window",
    yFormat: (v) => v.toFixed(2),
    series: [
      { name: "Growth model", short: "Model", color: SERIES.ochre, values: windows.map((w) => w.model) },
      { name: "EA potential baseline", short: "Potential", color: SERIES.blue, values: windows.map((w) => w.ea_potential) },
      { name: "No change", short: "No change", color: SERIES.magenta, values: windows.map((w) => w.persistence) },
    ],
    height: 250,
  });
}

const num = (name) => name.replace(/\D+/g, "");
