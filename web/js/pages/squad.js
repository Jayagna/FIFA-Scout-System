import { post } from "../api.js";
import { face, h, money, notice, pitchLines, reducedMotion, round, setTitle, signed } from "../ui.js";

export async function render(view, { meta, query, setQuery, isCurrent }) {
  setTitle("Squad builder");
  const leagueNames = meta.leagues.map((l) => l.name);
  const startLeague = leagueNames.includes(query.get("league")) ? query.get("league") : meta.default_league;
  const clubsOf = (lg) => meta.leagues.find((l) => l.name === lg)?.clubs || [];

  const f = {
    league: h("select", { id: "sq-league" }, leagueNames.map((n) => h("option", { selected: n === startLeague }, n))),
    club: h("select", { id: "sq-club" }),
    formation: h("select", { id: "sq-form" }, Object.keys(meta.formations).map((n) => h("option", { selected: n === (query.get("formation") || "4-3-3") }, n))),
    budget: h("input", { type: "number", id: "sq-budget", min: 0, max: 1500, step: 5, value: query.get("budget") || 50 }),
    wage: h("input", { type: "number", id: "sq-wage", min: 0, max: 50000, step: 10, placeholder: "Wage bill + 15%" }),
    signings: h("input", { type: "range", id: "sq-sign", min: 1, max: 6, value: 3 }),
    age: h("input", { type: "range", id: "sq-age", min: 17, max: 40, value: 40 }),
    objective: h("select", { id: "sq-obj" },
      h("option", { value: "next" }, `Next season (${meta.next_name} forecast)`),
      h("option", { value: "now" }, "Current ratings"),
      h("option", { value: "two" }, `Two seasons (${meta.next2_name} forecast)`)),
    sales: h("input", { type: "checkbox", id: "sq-sales" }),
    maxSales: h("input", { type: "range", id: "sq-maxsales", min: 1, max: 6, value: 3, disabled: true }),
    markup: h("input", { type: "range", id: "sq-markup", min: 1, max: 2, step: 0.05, value: 1.2 }),
    market: h("select", { id: "sq-market" }, h("option", { value: "" }, "Anywhere"), leagueNames.map((n) => h("option", n))),
  };
  const outs = { signings: h("output", "3"), age: h("output", "Any"), maxSales: h("output", "3"), markup: h("output", "1.20×") };
  const fillClubs = (pick) => {
    const clubs = clubsOf(f.league.value);
    f.club.replaceChildren(...clubs.map((c) => h("option", { selected: c === pick }, c)));
  };
  fillClubs(query.get("club"));

  const run = h("button.btn.primary", { type: "submit" }, "Find the best signings");
  const pitch = h("div.pitch", { html: pitchLines() });
  const score = h("div.scoreline");
  const status = h("p.small.muted", { "aria-live": "polite" });
  const deals = h("div.deals");

  view.append(h("div.wrap.page",
    h("header.page-head",
      h("h1", "Squad builder"),
      h("p", "Pick your club and budget. The planner checks every affordable combination of signings (and sales, if you allow them) and returns the strongest starting XI it can field, solved exactly.")),
    h("div.builder",
      h("form.panel", { onsubmit: (e) => { e.preventDefault(); solve(false); } },
        h("h2", "Your club"),
        field("League", f.league, "sq-league"),
        field("Club", f.club, "sq-club"),
        field("Formation", f.formation, "sq-form"),
        h("h2", { style: { marginTop: "6px" } }, "Your window"),
        h("div.two", field("Transfer budget (€M)", f.budget, "sq-budget"), field("Wages (€K a week)", f.wage, "sq-wage")),
        rangeField("Most signings", f.signings, outs.signings, "sq-sign"),
        rangeField("Oldest signing", f.age, outs.age, "sq-age"),
        field("Pick players for", f.objective, "sq-obj"),
        rangeField("Asking price, times value", f.markup, outs.markup, "sq-markup"),
        field("Buy from", f.market, "sq-market"),
        h("label.check", f.sales, "Sell players to raise money"),
        rangeField("Most sales", f.maxSales, outs.maxSales, "sq-maxsales"),
        run),
      h("section", { "aria-label": "Plan" },
        score, h("div.pitch-stage", pitch), status, deals))));

  f.league.addEventListener("change", () => { fillClubs(); f.wage.value = ""; solve(true); });
  f.club.addEventListener("change", () => { f.wage.value = ""; solve(true); });
  f.formation.addEventListener("change", () => solve(lastWasBaseline));
  f.signings.addEventListener("input", () => (outs.signings.textContent = f.signings.value));
  f.age.addEventListener("input", () => (outs.age.textContent = f.age.value >= 40 ? "Any" : f.age.value));
  f.maxSales.addEventListener("input", () => (outs.maxSales.textContent = f.maxSales.value));
  f.markup.addEventListener("input", () => (outs.markup.textContent = `${(+f.markup.value).toFixed(2)}×`));
  f.sales.addEventListener("change", () => (f.maxSales.disabled = !f.sales.checked));

  let seq = 0;
  let lastWasBaseline = true;

  async function solve(baseline) {
    const my = ++seq;
    lastWasBaseline = baseline;
    run.disabled = true;
    run.textContent = baseline ? "Loading squad…" : "Solving…";
    status.textContent = "";
    setQuery({ league: f.league.value, club: f.club.value, formation: f.formation.value, budget: f.budget.value });
    const body = {
      club: f.club.value, formation: f.formation.value,
      budget: baseline ? 0 : +f.budget.value || 0,
      wage_budget: f.wage.value === "" ? null : +f.wage.value,
      max_signings: baseline ? 1 : +f.signings.value,
      max_age: +f.age.value >= 40 ? null : +f.age.value,
      objective: f.objective.value, allow_sales: !baseline && f.sales.checked,
      max_sales: +f.maxSales.value, markup: +f.markup.value,
      market_leagues: f.market.value ? [f.market.value] : null,
    };
    try {
      const plan = await post("squad", body);
      if (my !== seq || !isCurrent()) return;
      if (plan.status !== "optimal") throw new Error(`No plan found (${plan.status}). Raise the wage budget or allow fewer signings.`);
      if (f.wage.value === "") f.wage.placeholder = String(Math.round(plan.wage_budget / 1e3));
      draw(plan, baseline);
    } catch (err) {
      if (my === seq) { deals.replaceChildren(notice(err)); }
    } finally {
      if (my === seq) { run.disabled = false; run.textContent = "Find the best signings"; }
    }
  }

  function draw(plan, baseline) {
    const label = { now: "XI rating now", next: `XI rating, ${meta.next_name}`, two: `XI rating, ${meta.next2_name}` }[plan.objective];
    score.replaceChildren(...[
      h("div.score", h("b", plan.xi_rating.toFixed(1)), h("span", label),
        baseline ? null : h("em", `${signed(plan.xi_rating - plan.baseline_xi_rating)} vs no signings`)),
      baseline ? null : h("div.score", h("b", money(plan.spend)), h("span", `spent of ${money(plan.budget)}`)),
      plan.income ? h("div.score", h("b", money(plan.income)), h("span", "raised from sales")) : null,
      h("div.score", h("b", money(plan.wage_bill)), h("span", `wages a week, cap ${money(plan.wage_budget)}`)),
    ].filter(Boolean));

    const prev = new Map([...pitch.querySelectorAll(".token")].map((t) => [t.dataset.slot, t]));
    const keep = new Set();
    for (const s of plan.xi) {
      const p = s.player;
      let tok = prev.get(s.slot);
      const changed = !tok || tok.dataset.id !== String(p.id);
      if (changed) {
        const fresh = h("a.token", { href: `#/player/${p.id}`, "data-slot": s.slot, "data-id": p.id, style: { "--c": `var(--${p.tier})` } },
          face(p), h("span.tag", h("b", Math.round(s.rating)), p.name), h("span.slot", s.slot.replace(/\d$/, "")));
        if (s.signing) fresh.classList.add("new");
        if (!reducedMotion() && tok) fresh.classList.add("arrive");
        tok ? tok.replaceWith(fresh) : pitch.append(fresh);
        tok = fresh;
      }
      tok.style.left = `${s.x}%`;
      tok.style.bottom = `${s.y * 0.94}%`;
      tok.setAttribute("aria-label", `${s.slot}: ${p.name}, ${Math.round(s.rating)} at this position${s.signing ? ", new signing" : ""}`);
      keep.add(s.slot);
    }
    for (const [slot, t] of prev) if (!keep.has(slot)) t.remove();

    status.textContent = baseline
      ? `${plan.club}'s strongest XI from the current squad. Ratings are each player's rating at that position${plan.objective === "now" ? "" : " plus his forecast change"}.`
      : `Best of ${plan.candidates.toLocaleString()} candidates left after pruning that cannot remove the optimum. Solved in ${plan.solve_seconds.toFixed(1)}s.`;

    if (baseline) { deals.replaceChildren(); return; }
    deals.replaceChildren(
      h("div", h("h2", { style: { fontSize: "var(--t-md)", margin: "0 0 10px" } }, `Sign (${plan.buys.length})`),
        plan.buys.length ? h("div.deal-list", plan.buys.map((p) => deal(p, "buy", meta))) : h("p.muted", "No signing improves the XI within these limits. Raise the budget or wage cap.")),
      h("div", h("h2", { style: { fontSize: "var(--t-md)", margin: "0 0 10px" } }, `Sell (${plan.sales.length})`),
        plan.sales.length ? h("div.deal-list", plan.sales.map((p) => deal(p, "sell", meta))) : h("p.muted", f.sales.checked ? "No sale makes the XI stronger." : "Sales are switched off.")));
  }

  solve(true);
}

function deal(p, kind, meta) {
  return h(`a.deal.${kind}`, { href: `#/player/${p.id}` },
    face(p),
    h("div", h("strong", p.name), h("small", `${p.pos}, ${round(p.age)}, ${kind === "buy" ? p.club : "your squad"}. Rated ${p.overall}, ${meta.next_name} ${round(p.pred_t1)}`)),
    h("div.fee", money(p.fee), h("small", kind === "buy" ? `${money(p.wage)}/wk` : "fee")));
}

function field(label, input, id) {
  return h("div.field", h("label", { for: id }, label), input);
}
function rangeField(label, input, out, id) {
  return h("div.field", h("label.row-label", { for: id }, label, out), input);
}
