import { get } from "../api.js";
import { lineChart, SERIES } from "../charts.js";
import { debounce, face, h, money, notice, pitchLines, playerCard, round, setTitle } from "../ui.js";

// where each position rating sits on the mini pitch (x across, y up from own goal)
const POS_XY = {
  GK: [50, 6], CB: [50, 20], LB: [14, 25], RB: [86, 25], LWB: [10, 41], RWB: [90, 41],
  CDM: [50, 36], CM: [50, 51], LM: [13, 57], RM: [87, 57], CAM: [50, 64],
  LW: [16, 79], RW: [84, 79], CF: [50, 77], ST: [50, 90],
};
const tierOf = (v) => (v >= 75 ? "gold" : v >= 65 ? "silver" : "bronze");
const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

export async function render(view, ctx) {
  if (!ctx.arg) return picker(view, ctx);
  const { meta } = ctx;
  let p;
  try {
    p = await get(`player/${ctx.arg}`);
  } catch (err) {
    view.append(h("div.wrap.page", notice(err)));
    return;
  }
  if (!ctx.isCurrent()) return;
  setTitle(p.name);
  const [f1, f2] = p.forecast;

  const hero = h("header.p-hero",
    h("div.p-face", { style: { "--c": `var(--${p.tier})` } }, face(p)),
    h("div",
      h("h1.p-name", p.long_name && p.long_name.length <= 26 ? p.long_name : p.name),
      h("p.p-sub",
        `${p.positions || p.pos}. ${round(p.age)} years old. `,
        p.club ? h("a", { href: `#/squad?league=${encodeURIComponent(p.league || "")}&club=${encodeURIComponent(p.club)}` }, p.club) : "Free agent",
        p.league ? `, ${p.league}` : "", p.nation ? `. ${p.nation}` : "")),
    h("div.p-numbers",
      num(p.overall, "Rating", p.edition_name, `tier-${p.tier}`),
      num(p.potential, "EA potential", "", ""),
      num(round(f1.pred), f1.name, f1.lo !== null ? `80% range ${round(f1.lo)}–${round(f1.hi)}` : "", "fc"),
      num(round(f2.pred), f2.name, f2.lo !== null ? `80% range ${round(f2.lo)}–${round(f2.hi)}` : "", "fc")));

  // rating history + forecast
  const labels = [...p.history.map((x) => short(x.name)), ...p.forecast.map((x) => short(x.name))];
  const nh = p.history.length;
  const pad = (arr, n) => arr.concat(Array(n).fill(null));
  const lead = Array(Math.max(0, nh - 1)).fill(null);
  const chart = lineChart({
    labels,
    ariaLabel: `${p.name}'s rating in every edition and the forecast for the next two`,
    yFormat: fmt,
    series: [
      { name: "Rating", color: SERIES.blue, values: pad(p.history.map((x) => x.overall), 2) },
      { name: "EA potential", short: "Potential", color: SERIES.magenta, dash: true, values: pad(p.history.map((x) => x.potential), 2) },
      { name: "Forecast", color: SERIES.ochre, values: [...lead, p.overall, f1.pred, f2.pred] },
    ],
    band: { name: "80% range", color: SERIES.ochre, lo: [...lead, p.overall, f1.lo, f2.lo], hi: [...lead, p.overall, f1.hi, f2.hi] },
    height: 280,
  });

  // position map
  const rated = p.positions_rated;
  const best = Math.max(...Object.keys(POS_XY).map((k) => rated[k] || 0));
  const map = h("div.pos-map", { html: pitchLines(), role: "img",
    "aria-label": `Rating at each position: ${Object.keys(POS_XY).map((k) => `${k} ${rated[k] ?? "n/a"}`).join(", ")}` });
  for (const [pos, [x, y]] of Object.entries(POS_XY)) {
    const v = rated[pos];
    if (v === null || v === undefined) continue;
    map.append(h(`div.pos-dot${v === best ? ".best" : ""}`, { style: { left: `${x}%`, bottom: `${y * 0.92}%`, "--c": `var(--${tierOf(v)})` } }, Math.round(v), h("small", pos)));
  }

  // attributes
  const groups = h("div.attr-groups", Object.entries(p.attributes).map(([g, items]) => h("div.attr-group",
    h("h3", g),
    items.map((a) => h("div.attr", h("span", a.label), h("b", a.value ?? "–"),
      h("div.bar", h(`i${a.value >= 80 ? ".hi" : a.value < 50 ? ".lo" : ""}`, { style: { width: `${a.value || 0}%` } })))))));

  const facts = h("dl.facts-list", [
    ["Height", p.height ? `${p.height} cm` : "–"], ["Weight", p.weight ? `${p.weight} kg` : "–"],
    ["Preferred foot", p.foot || "–"], ["Weak foot", p.weak_foot ? `${p.weak_foot} of 5` : "–"],
    ["Skill moves", p.skill_moves ? `${p.skill_moves} of 5` : "–"], ["Reputation", p.reputation ? `${p.reputation} of 5` : "–"],
    ["Value", money(p.value)], ["Wage", p.wage ? `${money(p.wage)} a week` : "–"],
    ["Release clause", money(p.release_clause)], ["Contract until", p.contract_end || "–"],
  ].map(([k, v]) => h("div", h("dt", k), h("dd", v))));

  view.append(h("div.wrap.page", { style: { paddingTop: "8px" } },
    hero,
    h("div.p-grid",
      h("section.box", h("h2", "Rating by edition"),
        h("p", `What EA rated him in each edition, and the model's forecast for ${f1.name} and ${f2.name} with its 80% range.`), chart),
      h("section.box", h("h2", "Rating at every position"),
        h("p", `The highest is filled in. The squad builder uses these, not the headline rating.`), map)),
    h("section.box", { style: { marginTop: "22px" } }, h("h2", "Attributes"), h("p", `${meta.edition_name} values. Gold bars are 80 or more.`), groups),
    h("section.box", { style: { marginTop: "22px" } }, h("h2", "Contract and profile"), facts),
    p.similar.length ? h("section.section",
      h("h2", "Plays like him"),
      h("p.lede", "The closest attribute profiles at other clubs, at the same position."),
      h("div.strip", p.similar.map((s) => playerCard(s, meta))),
      h("a.btn", { href: `#/replacements?id=${p.id}` }, `Find cheaper players like ${p.name}`)) : null));
}

function num(big, label, small, cls) {
  return h(`div.p-num${cls === "fc" ? ".fc" : ""}`, h(`b${cls && cls !== "fc" ? "." + cls : ""}`, big), h("span", label), small ? h("small", small) : null);
}
const short = (name) => name.replace("EA FC ", "FC ");

async function picker(view, { meta }) {
  setTitle("Player outlook");
  const input = h("input", { type: "search", id: "po-q", placeholder: "Search a player", autocomplete: "off" });
  const results = h("div.cards", { "aria-live": "polite" });
  view.append(h("div.wrap.page",
    h("header.page-head", h("h1", "Player outlook"),
      h("p", `Any player's rating history, the forecast for ${meta.next_name} and ${meta.next2_name}, his rating at every position and the players most like him.`)),
    h("div.field", { style: { maxWidth: "420px", marginBottom: "26px" } }, h("label", { for: "po-q" }, "Find a player"), input),
    results));
  const showTop = async () => {
    const r = await get("wonderkids", { max_age: 45, max_value: 2000, sort: "overall", limit: 12 });
    results.replaceChildren(...r.players.map((p) => playerCard(p, meta)));
  };
  input.addEventListener("input", debounce(async () => {
    const q = input.value.trim();
    if (q.length < 2) return showTop();
    const r = await get("search", { q, limit: 12 });
    results.replaceChildren(...(r.length ? r.map((p) => playerCard(p, meta)) : [h("p.muted", `No player called "${q}" in ${meta.edition_name}.`)]));
  }, 200));
  showTop();
}
