import { get } from "../api.js";
import { debounce, face, h, money, notice, playerCard, round, setTitle } from "../ui.js";

export async function render(view, { meta, query, setQuery, isCurrent }) {
  setTitle("Replacement finder");
  const box = h("div.search");
  const input = h("input", { type: "search", id: "rp-q", placeholder: "Search a player, e.g. Saka", autocomplete: "off",
    role: "combobox", "aria-expanded": "false", "aria-controls": "rp-list", "aria-autocomplete": "list" });
  const list = h("ul.suggest", { id: "rp-list", role: "listbox", hidden: true });
  box.append(input, list);
  const targetBox = h("div");
  const fee = h("input", { type: "number", id: "rp-fee", min: 0, max: 400, step: 1 });
  const age = h("input", { type: "range", id: "rp-age", min: 17, max: 40, value: +(query.get("max_age") || 40) });
  const ageOut = h("output", age.value >= 40 ? "Any" : age.value);
  const minOvr = h("input", { type: "number", id: "rp-min", min: 40, max: 99 });
  const same = h("input", { type: "checkbox", id: "rp-same", checked: query.get("same") !== "0" });
  const league = h("select", { id: "rp-league" }, h("option", { value: "" }, "Every league"), meta.leagues.map((l) => h("option", { selected: l.name === query.get("league") }, l.name)));
  const results = h("div", { "aria-live": "polite" });

  view.append(h("div.wrap.page",
    h("header.page-head",
      h("h1", "Replacement finder"),
      h("p", "Pick a player you'd like to replace or can't afford. The finder ranks everyone by how closely their detailed attributes match his, then keeps the ones under your price cap.")),
    h("div.finder",
      h("div.panel",
        h("div.field", h("label", { for: "rp-q" }, "Player to replace"), box),
        targetBox,
        h("div.two",
          h("div.field", h("label", { for: "rp-fee" }, "Max fee (€M)"), fee),
          h("div.field", h("label", { for: "rp-min" }, "Min rating"), minOvr)),
        h("div.field", h("label.row-label", { for: "rp-age" }, "Oldest", ageOut), age),
        h("div.field", h("label", { for: "rp-league" }, "League"), league),
        h("label.check", same, "Same position only")),
      results)));

  let target = null;
  let seq = 0;
  async function pick(id, keepFilters = false) {
    const p = await get(`player/${id}`);
    if (!isCurrent()) return;
    target = p;
    input.value = p.name;
    targetBox.replaceChildren(playerCard(p, meta));
    if (!keepFilters) {
      fee.value = Math.max(1, Math.round((p.value || 0) * 0.5 / 1e6));
      minOvr.value = Math.max(40, p.overall - 8);
    }
    find();
  }

  async function find() {
    if (!target) return;
    const my = ++seq;
    setQuery({ id: target.id, fee: fee.value, min: minOvr.value, max_age: age.value >= 40 ? "" : age.value,
               same: same.checked ? "" : "0", league: league.value });
    try {
      const r = await get("replacements", { id: target.id, max_fee: fee.value, min_overall: minOvr.value,
        max_age: age.value >= 40 ? null : age.value, same_position: same.checked ? 1 : 0, leagues: league.value ? [league.value] : null });
      if (my !== seq || !isCurrent()) return;
      if (!r.results.length) {
        results.replaceChildren(h("div.notice", h("strong", "Nobody matches these filters."), h("p", { style: { margin: "6px 0 0" } }, "Raise the fee cap or lower the minimum rating.")));
        return;
      }
      results.replaceChildren(
        h("p.small.muted", { style: { margin: "0 0 12px" } }, `${r.results.length} closest profiles to ${r.target.name} at ${money(r.max_fee)} or less. Similarity is the cosine of the standardised attribute profiles: 1 means identical.`),
        h("div.table-wrap", h("table.data",
          h("thead", h("tr", h("th", "Player"), h("th", "Pos"), h("th.num", "Age"), h("th", "Club"), h("th.num", "Rating"),
            h("th.num", meta.next_name), h("th", "Similarity"), h("th.num", "Fee"), h("th.num", "Saving"))),
          h("tbody", r.results.map((p) => h("tr",
            h("td", h("div.who", face(p), h("a", { href: `#/player/${p.id}` }, p.name))),
            h("td", p.pos), h("td.num", round(p.age)), h("td", p.club),
            h(`td.num.tier-${p.tier}`, h("b", p.overall)), h("td.num.fc", round(p.pred_t1)),
            h("td", h("div.sim", h("i", { style: { width: `${Math.max(4, (p.similarity - 0.5) * 2 * 110)}px` } }), h("span", p.similarity.toFixed(3)))),
            h("td.num", money(p.fee)), h("td.num.up", money(p.saving))))))));
    } catch (err) {
      results.replaceChildren(notice(err));
    }
  }

  // autocomplete
  let active = -1, items = [];
  const show = (arr) => {
    items = arr; active = -1;
    list.replaceChildren(...arr.map((p, i) => h("li", { role: "option", id: `rp-o${i}`, "aria-selected": "false",
      onmousedown: (e) => { e.preventDefault(); choose(i); } },
      face(p), h("div", h("div", p.name), h("small", `${p.pos}, ${p.club || "Free agent"}, ${p.overall}`)))));
    list.hidden = !arr.length;
    input.setAttribute("aria-expanded", String(!!arr.length));
  };
  const choose = (i) => { const p = items[i]; show([]); if (p) pick(p.id); };
  input.addEventListener("input", debounce(async () => {
    const q = input.value.trim();
    if (q.length < 2) return show([]);
    show(await get("search", { q, limit: 8 }));
  }, 160));
  input.addEventListener("keydown", (e) => {
    if (list.hidden) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      [...list.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
      input.setAttribute("aria-activedescendant", `rp-o${active}`);
    } else if (e.key === "Enter") { e.preventDefault(); choose(Math.max(0, active)); }
    else if (e.key === "Escape") show([]);
  });
  input.addEventListener("blur", () => setTimeout(() => show([]), 120));

  const later = debounce(find, 250);
  for (const el of [fee, minOvr]) el.addEventListener("input", later);
  age.addEventListener("input", () => { ageOut.textContent = age.value >= 40 ? "Any" : age.value; later(); });
  same.addEventListener("change", find);
  league.addEventListener("change", find);

  // start from the link's player, or the best player in the default league
  if (query.get("id")) {
    if (query.get("fee")) fee.value = query.get("fee");
    if (query.get("min")) minOvr.value = query.get("min");
    await pick(+query.get("id"), !!query.get("fee"));
  } else {
    // the highest-rated player in the default league, so the page never opens empty
    const top = await get("wonderkids", { max_age: 45, max_value: 2000, leagues: [meta.default_league], sort: "overall", limit: 1 });
    if (top.players[0]) await pick(top.players[0].id);
  }
}
