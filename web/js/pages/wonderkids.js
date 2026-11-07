import { get } from "../api.js";
import { debounce, face, h, money, notice, playerCard, round, setTitle } from "../ui.js";

const GROUPS = [["GK", "Keepers"], ["DEF", "Defenders"], ["MID", "Midfielders"], ["FWD", "Forwards"]];

export async function render(view, { meta, query, setQuery, isCurrent }) {
  setTitle("Wonderkids");
  const state = {
    max_age: +(query.get("max_age") || 21),
    max_value: +(query.get("max_value") || 15),
    groups: query.get("groups") ? query.get("groups").split("|") : GROUPS.map((g) => g[0]),
    league: query.get("league") || "",
    sort: query.get("sort") || "pred_t2",
    view: query.get("view") || "cards",
  };

  const ageOut = h("output", state.max_age);
  const valOut = h("output", money(state.max_value * 1e6));
  const age = h("input", { type: "range", min: 16, max: 30, value: state.max_age, id: "wk-age" });
  const val = h("input", { type: "range", min: 0, max: 120, step: 1, value: state.max_value, id: "wk-val" });
  const league = h("select", { id: "wk-league" }, h("option", { value: "" }, "Every league"),
    meta.leagues.map((l) => h("option", { value: l.name, selected: l.name === state.league }, l.name)));
  const sort = h("select", { id: "wk-sort" },
    [["pred_t2", `Forecast for ${meta.next2_name}`], ["growth", "Forecast growth"], ["potential", "EA potential"], ["overall", "Current rating"]]
      .map(([v, t]) => h("option", { value: v, selected: v === state.sort }, t)));
  const chips = h("div.chips", GROUPS.map(([g, label]) => h("label.chip",
    h("input", { type: "checkbox", value: g, checked: state.groups.includes(g) }), h("span", label))));

  const results = h("div", { "aria-live": "polite" });
  const count = h("p");
  const seg = h("div.seg", { role: "group", "aria-label": "Show as" },
    ["cards", "table"].map((v) => h("button", { type: "button", "aria-pressed": String(state.view === v), "data-v": v }, v === "cards" ? "Cards" : "Table")));

  view.append(h("div.wrap.page",
    h("header.page-head",
      h("h1", "Wonderkids"),
      h("p", `Young players ranked by the rating the model expects EA to give them in ${meta.next2_name}, with an 80% range. Set an age and a price you can afford.`)),
    h("form.filters", { onsubmit: (e) => e.preventDefault() },
      h("div.field", h("label.row-label", { for: "wk-age" }, "Maximum age", ageOut), age),
      h("div.field", h("label.row-label", { for: "wk-val" }, "Maximum value", valOut), val),
      h("fieldset.field", h("legend", "Positions"), chips),
      h("div.field", h("label", { for: "wk-league" }, "League"), league),
      h("div.field", h("label", { for: "wk-sort" }, "Rank by"), sort)),
    h("div.result-bar", count, seg),
    results));

  let seq = 0;
  const load = async () => {
    const my = ++seq;
    setQuery({ max_age: state.max_age, max_value: state.max_value, groups: state.groups.length === 4 ? "" : state.groups,
               league: state.league, sort: state.sort, view: state.view === "cards" ? "" : state.view });
    results.setAttribute("aria-busy", "true");
    try {
      const r = await get("wonderkids", { max_age: state.max_age, max_value: state.max_value,
        groups: state.groups, leagues: state.league ? [state.league] : null, sort: state.sort, limit: 60 });
      if (my !== seq || !isCurrent()) return;
      count.textContent = r.total
        ? `${r.total.toLocaleString()} players match. Showing the top ${Math.min(60, r.total)}.`
        : "";
      if (!r.players.length) {
        results.replaceChildren(h("div.notice", h("strong", "Nobody matches these filters."), h("p", { style: { margin: "6px 0 0" } }, "Raise the maximum value or age, or add more positions.")));
      } else if (state.view === "cards") {
        results.replaceChildren(h("div.cards", r.players.map((p) => playerCard(p, meta))));
      } else {
        results.replaceChildren(table(r.players, meta));
      }
    } catch (err) {
      results.replaceChildren(notice(err));
    } finally {
      results.removeAttribute("aria-busy");
    }
  };
  const later = debounce(load, 220);

  age.addEventListener("input", () => { state.max_age = +age.value; ageOut.textContent = age.value; later(); });
  val.addEventListener("input", () => { state.max_value = +val.value; valOut.textContent = money(state.max_value * 1e6); later(); });
  league.addEventListener("change", () => { state.league = league.value; load(); });
  sort.addEventListener("change", () => { state.sort = sort.value; load(); });
  chips.addEventListener("change", () => {
    state.groups = [...chips.querySelectorAll("input:checked")].map((i) => i.value);
    load();
  });
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.view = b.dataset.v;
    seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    load();
  });
  load();
}

export function table(players, meta) {
  return h("div.table-wrap", h("table.data",
    h("thead", h("tr",
      h("th", "Player"), h("th", "Pos"), h("th.num", "Age"), h("th", "Club"),
      h("th.num", "Rating"), h("th.num", "Potential"),
      h("th.num", meta.next_name), h("th.num", meta.next2_name),
      h("th.num", "Value"), h("th.num", "Wage / wk"))),
    h("tbody", players.map((p) => h("tr",
      h("td", h("div.who", face(p), h("a", { href: `#/player/${p.id}` }, p.name))),
      h("td", p.pos), h("td.num", round(p.age)), h("td", p.club || "Free agent"),
      h(`td.num.tier-${p.tier}`, h("b", p.overall)), h("td.num", p.potential),
      h("td.num", h("span.fc", round(p.pred_t1)), " ", h("span.range", `${round(p.pred_t1_lo)}–${round(p.pred_t1_hi)}`)),
      h("td.num", h("span.fc", round(p.pred_t2)), " ", h("span.range", `${round(p.pred_t2_lo)}–${round(p.pred_t2_hi)}`)),
      h("td.num", money(p.value)), h("td.num", money(p.wage)))))));
}
