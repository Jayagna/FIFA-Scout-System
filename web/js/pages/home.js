import { get } from "../api.js";
import { h, playerCard, reducedMotion, setTitle } from "../ui.js";
import { rollingChart } from "../charts.js";

export async function render(view, { meta }) {
  setTitle("");
  const canvas = h("canvas", { "aria-hidden": "true" });
  const hero = h("section.hero", { "aria-labelledby": "hero-title" },
    h("div.stage",
      h("div.fallback-pitch"),
      canvas,
      h("p.hero-caption", `The under-21 XI with the highest ratings forecast for ${meta.next2_name}. Click a player to open his outlook.`)),
    h("div.hero-copy", h("div.wrap",
      h("h1#hero-title", "Sign them before they're stars."),
      h("p", `Forecasts of the ratings EA will give every player in ${meta.next_name} and ${meta.next2_name}, a transfer planner that finds the best signings for your budget exactly, and a finder for cheaper players with the same profile. Built on ${meta.player_seasons.toLocaleString()} player-seasons from FIFA 15 to ${meta.edition_name}.`),
      h("div.ctas",
        h("a.btn.primary", { href: "#/wonderkids" }, "Find wonderkids"),
        h("a.btn", { href: "#/squad" }, "Plan a transfer window")))));

  const risersBox = h("div.strip", { "aria-busy": "true" });
  const evidence = h("div.evidence");

  view.append(hero,
    h("div.wrap",
      h("section.section",
        h("h2", `Top of the class in ${meta.next2_name}`),
        h("p.lede", `Under-21s ranked by the rating the model expects EA to give them two editions from now. Each card shows today's rating, the forecasts, and EA's own potential number.`),
        risersBox),
      h("section.section",
        h("h2", "Three tools"),
        h("p.lede", "Each answers one question you ask before a transfer window."),
        h("div.tools",
          tool("#/wonderkids", "Wonderkid finder", "Who will be rated highest in two years, for the money I have?", "Rank young players"),
          tool("#/squad", "Squad builder", "Who should I buy, and sell, to field the best XI on this budget and wage bill?", "Build a squad"),
          tool("#/replacements", "Replacement finder", "Who plays like my star but costs half as much?", "Find look-alikes"))),
      h("section.section",
        h("h2", "How it's tested"),
        h("p.lede", "Every number here comes from editions the models never saw while training."),
        evidence)));

  // 3D pitch, loaded only on this page
  let scene = null;
  let alive = true;
  (async () => {
    try {
      const xi = await get("dream-xi");
      const { mountPitch } = await import("../scene/pitch.js");
      if (!alive) return;
      scene = await mountPitch(canvas, xi, { reduced: reducedMotion(), onPick: (id) => { location.hash = `#/player/${id}`; } });
      if (!alive) scene.dispose();
      hero.querySelector(".fallback-pitch").remove();
    } catch (err) {
      console.warn("3D pitch unavailable, showing the flat one", err);
      canvas.remove();
    }
  })();

  get("risers", { limit: 12 }).then((list) => {
    risersBox.replaceChildren(...list.map((p) => playerCard(p, meta)));
    risersBox.removeAttribute("aria-busy");
  });

  get("report").then((r) => evidence.replaceChildren(...evidenceBlocks(r, meta))).catch(() => {});

  return () => { alive = false; scene?.dispose(); };
}

function tool(href, title, q, go) {
  return h("a.tool", { href }, h("h3", title), h("p", q), h("span.go", go));
}

function evidenceBlocks(r, meta) {
  const f2 = r.forecast?.h2, f1 = r.forecast?.h1;
  const facts = [];
  if (f2) {
    const gain = 1 - f2.model / f2.ea_potential;
    facts.push(fact(`${(gain * 100).toFixed(0)}%`, `smaller error than EA's potential number, two editions ahead (${f2.test}: ${f2.model.toFixed(2)} vs ${f2.ea_potential.toFixed(2)} rating points)`));
    if (f2.risers_hits_model)
      facts.push(fact(`${Math.round(f2.risers_hits_model * 100)} of 100`, `of the model's top under-23 picks became top-10% risers, against ${Math.round(f2.risers_hits_potential * 100)} picking by EA's potential and ${Math.round(f2.risers_base_rate * 100)} by chance`));
  }
  const o = r.optimizer;
  if (o) facts.push(fact(`+${(o.extra_gain_vs_greedy_pct * 100).toFixed(0)}%`, `more XI improvement from the same money than buying the biggest affordable upgrade each time (${o.scenarios} Premier League scenarios, never worse)`));
  const s = r.similarity?.cheaper_lookalikes_pl_80plus;
  if (s) facts.push(fact(`${Math.round(s.found_share * 100)}%`, `of Premier League players rated 80+ have a same-position look-alike within 3 points at half the price or less`));

  const chartBox = h("div.chart-box",
    h("h3", "Error one edition ahead, every test window"),
    h("p", "Mean absolute error in rating points; each window uses a model trained only on earlier editions. Lower is better."),
    h("div.rolling"));
  if (r.rolling?.h1?.length) chartBox.querySelector(".rolling").replaceChildren(rollingChart(r.rolling.h1));
  else chartBox.remove();
  const prev = meta.editions.find((e) => e.edition === meta.edition - 1);
  if (f1 && prev && f1.persistence < f1.model) {
    chartBox.append(h("p.small.muted", { style: { margin: "10px 0 0" } },
      `The last window is the exception: the ${prev.name} file is a late-season roster with that season's upgrades already applied, so "no change" is hard to beat one step ahead (${f1.persistence.toFixed(2)} vs ${f1.model.toFixed(2)}). Two editions ahead, both files are launch rosters and the model leads.`));
  }
  return [h("div.facts", facts), chartBox];
}

function fact(big, text) {
  return h("div.fact", h("b", big), h("span", text));
}
