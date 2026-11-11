import { get, post } from "../api.js";
import { face, h, money, notice, pitchLines, reducedMotion, round, setTitle, signed } from "../ui.js";

const MODES = [
  ["now", "Win now", "Today's ratings. Veterans are cheap for what they give you this season."],
  ["balanced", "Balanced", "Value over the next two seasons, counting the chance each player retires or leaves the game."],
  ["future", "Build for the future", "Ratings two seasons out, with the same retirement risk. Favours players still improving."],
];

export async function render(view, { meta, query, setQuery, isCurrent }) {
  setTitle("Squad builder");
  const leagueNames = meta.leagues.map((l) => l.name);
  const startLeague = leagueNames.includes(query.get("league")) ? query.get("league") : meta.default_league;
  const clubsOf = (lg) => meta.leagues.find((l) => l.name === lg)?.clubs || [];
  const hasRumours = (meta.rumours?.links || 0) > 0;

  const state = {
    mode: MODES.some((m) => m[0] === query.get("mode")) ? query.get("mode") : "balanced",
    ban: new Map(), lock: new Map(),          // id -> name
    plans: [], active: 0, seed: null,
  };

  const f = {
    league: h("select", { id: "sq-league" }, leagueNames.map((n) => h("option", { selected: n === startLeague }, n))),
    club: h("select", { id: "sq-club" }),
    formation: h("select", { id: "sq-form" }, Object.keys(meta.formations).map((n) => h("option", { selected: n === (query.get("formation") || "4-3-3") }, n))),
    budget: h("input", { type: "number", id: "sq-budget", min: 0, max: 1500, step: 5, value: query.get("budget") || 50 }),
    wage: h("input", { type: "number", id: "sq-wage", min: 0, max: 50000, step: 10, placeholder: "Wage bill + 15%" }),
    signings: h("input", { type: "range", id: "sq-sign", min: 1, max: 6, value: 3 }),
    age: h("input", { type: "range", id: "sq-age", min: 17, max: 40, value: 40 }),
    sales: h("input", { type: "checkbox", id: "sq-sales" }),
    maxSales: h("input", { type: "range", id: "sq-maxsales", min: 1, max: 6, value: 3, disabled: true }),
    markup: h("input", { type: "range", id: "sq-markup", min: 1, max: 2, step: 0.05, value: 1.2 }),
    market: h("select", { id: "sq-market" }, h("option", { value: "" }, "Anywhere"), leagueNames.map((n) => h("option", n))),
    realistic: h("input", { type: "checkbox", id: "sq-real", checked: query.get("realistic") !== "0" }),
    rumours: h("input", { type: "checkbox", id: "sq-rum", checked: hasRumours && query.get("rumours") === "1", disabled: !hasRumours }),
  };
  const outs = { signings: h("output", "3"), age: h("output", "Any"), maxSales: h("output", "3"), markup: h("output", "1.20×") };
  const fillClubs = (pick) => {
    const clubs = clubsOf(f.league.value);
    f.club.replaceChildren(...clubs.map((c) => h("option", { selected: c === pick }, c)));
  };
  fillClubs(query.get("club"));

  const modeHelp = h("p.small.muted.mode-help");
  const modes = h("div.seg.modes", { role: "group", "aria-label": "Build mode" },
    MODES.map(([v, label]) => h("button", { type: "button", "data-v": v, "aria-pressed": String(v === state.mode) }, label)));
  const setMode = (v) => {
    state.mode = v;
    modes.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
    modeHelp.textContent = MODES.find((m) => m[0] === v)[2];
  };
  setMode(state.mode);

  const run = h("button.btn.primary", { type: "submit" }, "Find the best signings");
  const shuffle = h("button.btn", { type: "button", title: "A different good plan each time: picks at random among plans within 0.4 XI points of the best" }, "Shuffle");
  const picks = h("div.picks", { "aria-live": "polite" });
  const tabs = h("div.plan-tabs", { role: "tablist", "aria-label": "Plans" });
  const pitch = h("div.pitch", { html: pitchLines() });
  const score = h("div.scoreline");
  const status = h("p.small.muted", { "aria-live": "polite" });
  const deals = h("div.deals");
  const rumourBox = h("section.rumour-box", { "aria-label": "Transfer rumours" });

  view.append(h("div.wrap.page",
    h("header.page-head",
      h("h1", "Squad builder"),
      h("p", "Pick your club and budget. The planner checks every affordable combination of signings (and sales, if you allow them) and returns the strongest starting XI it can field, plus two different alternatives.")),
    h("div.builder",
      h("form.panel", { onsubmit: (e) => { e.preventDefault(); state.seed = null; solve(false); } },
        h("h2", "Your club"),
        field("League", f.league, "sq-league"),
        field("Club", f.club, "sq-club"),
        field("Formation", f.formation, "sq-form"),
        h("h2", { style: { marginTop: "6px" } }, "Build for"),
        modes, modeHelp,
        h("h2", { style: { marginTop: "6px" } }, "Your window"),
        h("div.two", field("Transfer budget (€M)", f.budget, "sq-budget"), field("Wages (€K a week)", f.wage, "sq-wage")),
        rangeField("Most signings", f.signings, outs.signings, "sq-sign"),
        rangeField("Oldest signing", f.age, outs.age, "sq-age"),
        rangeField("Asking price, times value", f.markup, outs.markup, "sq-markup"),
        field("Buy from", f.market, "sq-market"),
        h("label.check", f.realistic, "Only players who'd realistically join"),
        h("label.check", f.rumours, hasRumours ? "Favour players linked by transfer rumours" : "Favour rumoured players (no rumour data yet)"),
        h("label.check", f.sales, "Sell players to raise money"),
        rangeField("Most sales", f.maxSales, outs.maxSales, "sq-maxsales"),
        picks,
        h("div.btn-row", run, shuffle)),
      h("section", { "aria-label": "Plan" },
        score, tabs, h("div.pitch-stage", pitch), status, deals, rumourBox))));

  modes.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { setMode(b.dataset.v); if (!lastWasBaseline) solve(false); } });
  f.league.addEventListener("change", () => { fillClubs(); clubChanged(); });
  f.club.addEventListener("change", clubChanged);
  f.formation.addEventListener("change", () => solve(lastWasBaseline));
  f.signings.addEventListener("input", () => (outs.signings.textContent = f.signings.value));
  f.age.addEventListener("input", () => (outs.age.textContent = f.age.value >= 40 ? "Any" : f.age.value));
  f.maxSales.addEventListener("input", () => (outs.maxSales.textContent = f.maxSales.value));
  f.markup.addEventListener("input", () => (outs.markup.textContent = `${(+f.markup.value).toFixed(2)}×`));
  f.sales.addEventListener("change", () => (f.maxSales.disabled = !f.sales.checked));
  shuffle.addEventListener("click", () => { state.seed = Math.floor(Math.random() * 1e9); solve(false); });
  tabs.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.active = +b.dataset.i;
    draw(currentPlan, false);
  });

  function clubChanged() {
    f.wage.value = "";
    state.ban.clear(); state.lock.clear(); drawPicks();
    loadRumours();
    solve(true);
  }

  let seq = 0;
  let lastWasBaseline = true;
  let currentPlan = null;

  async function solve(baseline) {
    const my = ++seq;
    lastWasBaseline = baseline;
    run.disabled = shuffle.disabled = true;
    run.textContent = baseline ? "Loading squad…" : "Solving…";
    status.textContent = "";
    setQuery({ league: f.league.value, club: f.club.value, formation: f.formation.value, budget: f.budget.value,
               mode: state.mode === "balanced" ? "" : state.mode, realistic: f.realistic.checked ? "" : "0",
               rumours: f.rumours.checked ? "1" : "" });
    const body = {
      club: f.club.value, formation: f.formation.value,
      budget: baseline ? 0 : +f.budget.value || 0,
      wage_budget: f.wage.value === "" ? null : +f.wage.value,
      max_signings: baseline ? 1 : +f.signings.value,
      max_age: +f.age.value >= 40 ? null : +f.age.value,
      objective: state.mode, allow_sales: !baseline && f.sales.checked,
      max_sales: +f.maxSales.value, markup: +f.markup.value,
      market_leagues: f.market.value ? [f.market.value] : null,
      realistic: f.realistic.checked, rumours: f.rumours.checked,
      alternatives: baseline ? 0 : (state.seed === null ? 2 : 0), seed: baseline ? null : state.seed,
      ban_ids: baseline ? [] : [...state.ban.keys()], lock_ids: baseline ? [] : [...state.lock.keys()],
    };
    try {
      const plan = await post("squad", body);
      if (my !== seq || !isCurrent()) return;
      if (plan.status !== "optimal") throw new Error(`No plan found (${plan.status}). Raise the budget or wage cap, allow fewer signings, or unlock a player.`);
      if (f.wage.value === "") f.wage.placeholder = String(Math.round(plan.wage_budget / 1e3));
      currentPlan = plan;
      state.active = 0;
      draw(plan, baseline);
    } catch (err) {
      if (my === seq) { deals.replaceChildren(notice(err)); }
    } finally {
      if (my === seq) { run.disabled = shuffle.disabled = false; run.textContent = "Find the best signings"; }
    }
  }

  function draw(plan, baseline) {
    const plans = plan.plans?.length ? plan.plans : [plan];
    const p = plans[Math.min(state.active, plans.length - 1)];
    const label = {
      now: "XI rating now", next: `Expected XI rating, ${meta.next_name}`,
      balanced: "Expected XI rating, next two seasons", future: `Expected XI rating, ${meta.next2_name}`,
    }[plan.objective];
    score.replaceChildren(...[
      h("div.score", h("b", p.xi_rating.toFixed(1)), h("span", label),
        baseline ? null : h("em", `${signed(p.xi_rating - plan.baseline_xi_rating)} vs no signings`)),
      baseline ? null : h("div.score", h("b", money(p.spend)), h("span", `spent of ${money(plan.budget)}`)),
      p.income ? h("div.score", h("b", money(p.income)), h("span", "raised from sales")) : null,
      h("div.score", h("b", money(p.wage_bill)), h("span", `wages a week, cap ${money(plan.wage_budget)}`)),
    ].filter(Boolean));

    tabs.replaceChildren(...(baseline || plans.length < 2 ? [] : plans.map((q, i) =>
      h("button", { type: "button", role: "tab", "data-i": i, "aria-selected": String(i === state.active) },
        h("b", `Plan ${"ABC"[i]}`), h("span", q.xi_rating.toFixed(2)),
        h("small", q.buys.map((b) => b.name).join(", ") || "no signings")))));

    const prev = new Map([...pitch.querySelectorAll(".token")].map((t) => [t.dataset.slot, t]));
    const keep = new Set();
    for (const s of p.xi) {
      const pl = s.player;
      let tok = prev.get(s.slot);
      if (!tok || tok.dataset.id !== String(pl.id)) {
        const fresh = h("a.token", { href: `#/player/${pl.id}`, "data-slot": s.slot, "data-id": pl.id, style: { "--c": `var(--${pl.tier})` } },
          face(pl), h("span.tag", h("b", Math.round(s.rating_if_stays ?? s.rating)), pl.name), h("span.slot", s.slot.replace(/\d$/, "")));
        if (s.signing) fresh.classList.add("new");
        if (!reducedMotion() && tok) fresh.classList.add("arrive");
        tok ? tok.replaceWith(fresh) : pitch.append(fresh);
        tok = fresh;
      }
      tok.style.left = `${s.x}%`;
      tok.style.bottom = `${s.y * 0.94}%`;
      const stay = s.stay < 0.995 ? `, ${Math.round(s.stay * 100)}% chance still in the game` : "";
      tok.setAttribute("aria-label", `${s.slot}: ${pl.name}, ${Math.round(s.rating_if_stays ?? s.rating)} at this position${stay}${s.signing ? ", new signing" : ""}`);
      keep.add(s.slot);
    }
    for (const [slot, t] of prev) if (!keep.has(slot)) t.remove();

    const extras = [];
    if (plan.realism_dropped) extras.push(`${plan.realism_dropped.toLocaleString()} players at much stronger clubs left out`);
    if (plan.rumour_links) extras.push(`${plan.rumour_links} rumoured target${plan.rumour_links === 1 ? "" : "s"} favoured`);
    if (state.seed !== null && !baseline) extras.push("Shuffled: a near-best plan picked at random");
    status.textContent = baseline
      ? `${plan.club}'s strongest XI from the current squad. Numbers on the pitch are each player's rating at that position${plan.objective === "now" ? "" : " in the season you're planning for"}.`
      : [`Best of ${p.candidates.toLocaleString()} candidates, solved exactly in ${plan.solve_seconds.toFixed(1)}s`, ...extras, ...(p.notes || [])].join(". ") + ".";

    if (baseline) { deals.replaceChildren(); return; }
    const actions = (pl) => h("div.deal-actions",
      h("button.mini", { type: "button", onclick: () => { state.lock.delete(pl.id); state.ban.set(pl.id, pl.name); drawPicks(); solve(false); } }, "Not him"),
      h("button.mini", { type: "button", "aria-pressed": String(state.lock.has(pl.id)), onclick: () => { state.ban.delete(pl.id); state.lock.set(pl.id, pl.name); drawPicks(); solve(false); } }, state.lock.has(pl.id) ? "Kept" : "Keep him"));
    deals.replaceChildren(
      h("div", h("h2", { style: { fontSize: "var(--t-md)", margin: "0 0 10px" } }, `Sign (${p.buys.length})`),
        p.buys.length ? h("div.deal-list", p.buys.map((b) => deal(b, "buy", meta, plan.objective, actions(b))))
          : h("p.muted", "No signing improves the XI within these limits. Raise the budget or wage cap.")),
      h("div", h("h2", { style: { fontSize: "var(--t-md)", margin: "0 0 10px" } }, `Sell (${p.sales.length})`),
        p.sales.length ? h("div.deal-list", p.sales.map((b) => deal(b, "sell", meta, plan.objective)))
          : h("p.muted", f.sales.checked ? "No sale makes the XI stronger." : "Sales are switched off.")));
  }

  function drawPicks() {
    const chip = (map, id, name, verb) => h("span.pick", `${verb} ${name}`,
      h("button", { type: "button", "aria-label": `Remove ${name}`, onclick: () => { map.delete(id); drawPicks(); } }, "×"));
    const items = [...[...state.lock].map(([id, n]) => chip(state.lock, id, n, "Keep")),
                   ...[...state.ban].map(([id, n]) => chip(state.ban, id, n, "Not"))];
    picks.replaceChildren(...items);
  }

  async function loadRumours() {
    rumourBox.replaceChildren();
    if (!hasRumours) return;
    try {
      const r = await get("rumours", { club: f.club.value });
      if (!isCurrent() || r.club !== f.club.value) return;
      if (!r.players.length) {
        rumourBox.append(h("h2", `Linked with ${r.club}`), h("p.muted.small", `No transfer rumours about ${r.club} in the last feed (${r.generated?.slice(0, 10) || "unknown date"}).`));
        return;
      }
      rumourBox.append(
        h("h2", `Linked with ${r.club}`),
        h("p.muted.small", `From headlines up to ${r.generated?.slice(0, 10)}. Tick "Favour players linked by transfer rumours" to give them a boost, or keep one to force him in.`),
        h("div.deal-list", r.players.map((pl) => h("div.deal.rumour",
          face(pl),
          h("div",
            h("a", { href: `#/player/${pl.id}` }, h("strong", pl.name)),
            h("small", `${pl.pos}, ${round(pl.age)}, ${pl.club}. ${stageText(pl.rumour.stage)}`),
            h("small.src", ...pl.rumour.articles.slice(0, 2).map((a, i) => [i ? " · " : "", h("a", { href: a.url, target: "_blank", rel: "noopener" }, `${a.source}, ${a.date}`)]))),
          h("div.deal-actions",
            h("button.mini", { type: "button", onclick: () => { state.ban.delete(pl.id); state.lock.set(pl.id, pl.name); drawPicks(); solve(false); } }, "Keep him"))))));
    } catch { /* rumours are optional */ }
  }

  drawPicks();
  loadRumours();
  solve(true);
}

function stageText(stage) {
  return { 3: "Deal reported close or agreed.", 2: "Bid or talks reported.", 1: "Interest reported." }[stage] || "Linked.";
}

function deal(p, kind, meta, objective, actions) {
  const stay = objective === "now" ? "" :
    (p.stay_t1 != null ? ` Still in the game: ${Math.round(p.stay_t1 * 100)}% next season, ${Math.round(p.stay_t2 * 100)}% in two.` : "");
  const rumour = p.rumour ? h("small.src", `Rumoured: `, h("a", { href: p.rumour.articles[0].url, target: "_blank", rel: "noopener" }, `${p.rumour.articles[0].source}, ${p.rumour.articles[0].date}`)) : null;
  return h(`div.deal.${kind}`,
    face(p),
    h("div",
      h("a", { href: `#/player/${p.id}` }, h("strong", p.name)),
      h("small", `${p.pos}, ${round(p.age)}, ${kind === "buy" ? p.club : "your squad"}. Rated ${p.overall}, ${meta.next_name} ${round(p.pred_t1)}, ${meta.next2_name} ${round(p.pred_t2)}.${kind === "buy" ? stay : ""}`),
      rumour, actions || null),
    h("div.fee", money(p.fee), h("small", kind === "buy" ? `${money(p.wage)}/wk` : "fee")));
}

function field(label, input, id) {
  return h("div.field", h("label", { for: id }, label), input);
}
function rangeField(label, input, out, id) {
  return h("div.field", h("label.row-label", { for: id }, label, out), input);
}
