// Tests the pure JavaScript of /workspace/fleet/projects/o-lens/site/compare.html against compare.py / qualfmt.py.
//   /root/node-v22.23.4-linux-x64/bin/node scripts/qual/test_page.js
// Extracts the code between "// BEGIN PURE" and "// END PURE", runs it in a vm context (no DOM), and for every
// lens pair of every model in site/data/compare/index.json, for layer best + one fixed layer and k 1 / 10, checks that
// counts, per-task / difficulty / next-word counts, token statistics and recurring tokens equal Python's
// (python3 subprocess). Also runs the fake probe dumps (runs/qual/test/, scripts/qual/selftest.py) through the same
// checks, filterRows / sortRows, and renders every item (renderItem with all layers open) to catch exceptions
// and check hit / piece / near marks and HTML escaping. Not covered: real browser layout, fetch, event handlers.
// Exit code 0 = all passed.
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path"), { execFileSync } = require("child_process");
const SITE = "/workspace/fleet/projects/o-lens/site", R = "/workspace/o-lens-improvement";
const html = fs.readFileSync(SITE + "/compare.html", "utf8");
const src = html.slice(html.indexOf("// BEGIN PURE"), html.indexOf("// END PURE"));
const ctx = {}; vm.createContext(ctx); vm.runInContext(src + "\nthis.API={view,pairRows,counts,groupCounts,charStats,filterRows,sortRows,renderItem,tokClasses,esc};", ctx);
const P = ctx.API;
let nChecks = 0, fails = 0;
function eq(label, a, b) {
  nChecks++;
  const sa = JSON.stringify(a), sb = JSON.stringify(b);
  if (sa !== sb) { fails++; if (fails < 15) console.log("FAIL", label, "\n  js ", sa.slice(0, 400), "\n  py ", sb.slice(0, 400)); }
}
const PY = `
import json, sys
sys.path.insert(0, "${R}/scripts/qual")
import compare, qualfmt as Q
pa, pb, which, layer, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
A, B = Q.load(pa), Q.load(pb)
rows = compare.pair(A, B, which, layer, k)
kShow = min(10, A["meta"].get("k_top", 10), B["meta"].get("k_top", 10))
sa = Q.char_stats([r["a"] for r in rows], A["meta"], layer, kShow, 15)
sb = Q.char_stats([r["b"] for r in rows], B["meta"], layer, kShow, 15)
r6 = lambda s: dict(n_items=s["n_items"], slots=s["slots"], shares={k: round(v, 6) for k, v in s["shares"].items()},
                    recurring=[list(x) for x in s["recurring"]])
print(json.dumps(dict(counts=compare.counts(rows), task=compare.by(rows, "task"), diff=compare.by(rows, "difficulty"),
                      nxt=compare.by(rows, "is_next"), cls=[r["cls"] for r in rows], sa=r6(sa), sb=r6(sb))))
`;
function r6(s) { return { n_items: s.n_items, slots: s.slots, shares: Object.fromEntries(Object.entries(s.shares).map(([k, v]) => [k, Math.round(v * 1e6) / 1e6])), recurring: s.recurring }; }
function check(fa, fb, set, layer, k) {
  const A = JSON.parse(fs.readFileSync(fa, "utf8")), B = JSON.parse(fs.readFileSync(fb, "utf8"));
  const py = JSON.parse(execFileSync("python3", ["-c", PY, fa, fb, set, String(layer), String(k)], { encoding: "utf8" }));
  const rows = P.pairRows(A, B, set, String(layer), k);
  const kShow = Math.min(10, A.meta.k_top || 10, B.meta.k_top || 10);
  const tag = `${path.basename(fa)} vs ${path.basename(fb)} set=${set} layer=${layer} k=${k}`;
  eq(tag + " counts", P.counts(rows), py.counts);
  eq(tag + " cls", rows.map(r => r.cls), py.cls);
  eq(tag + " task", P.groupCounts(rows, "task"), py.task);
  eq(tag + " diff", P.groupCounts(rows, "difficulty"), py.diff);
  eq(tag + " next", P.groupCounts(rows, "is_next"), py.nxt);
  eq(tag + " statsA", r6(P.charStats(rows.map(r => r.a), A.meta, String(layer), kShow, 15)), py.sa);
  eq(tag + " statsB", r6(P.charStats(rows.map(r => r.b), B.meta, String(layer), kShow, 15)), py.sb);
  // filters partition the rows; sort keeps all rows
  const c = P.counts(rows);
  for (const cls of ["improved", "worsened", "both-hit", "both-miss", "control", "unscorable"])
    eq(tag + " filter " + cls, P.filterRows(rows, { cls }).length, c[cls]);
  eq(tag + " sort size", P.sortRows(rows).length, rows.length);
  // render every item with all layers open
  for (const r of rows) {
    const h = P.renderItem(r, A.meta.lens, B.meta.lens, kShow, true, A.meta, B.meta);
    nChecks++;
    const nHit = (h.match(/class="t hit/g) || []).length;
    const want = A.meta.layers.reduce((s, L, i) => s + (r.a.layers[i].h || []).filter(x => x < kShow).length, 0) +
      B.meta.layers.reduce((s, L, i) => s + (r.b.layers[i].h || []).filter(x => x < kShow).length, 0) +
      (r.ea.h || []).filter(x => x < kShow).length + (r.eb.h || []).filter(x => x < kShow).length;
    if (nHit !== want) { fails++; console.log("FAIL hit marks", r.a.id, nHit, want); }
    if (/<script|<img/i.test(h)) { fails++; console.log("FAIL unescaped", r.a.id); }
  }
  return rows.length;
}
const idx = JSON.parse(fs.readFileSync(SITE + "/data/compare/index.json", "utf8"));
let nPairs = 0;
for (const [tag, m] of Object.entries(idx.models)) {
  const files = m.lenses.map(l => `${SITE}/data/compare/${l.file}`);
  const fixed = String(m.lenses[0].layers[3]);
  for (const fa of files) for (const fb of files) {
    if (fa === fb) continue;
    for (const layer of ["best", fixed]) for (const k of [1, 10]) { check(fa, fb, "all", layer, k); nPairs++; }
  }
}
// fake probe dumps: controls, unscorable, piece hits, near marks, is_next
const fa = R + "/runs/qual/test/fake_a.json", fb = R + "/runs/qual/test/fake_b.json";
if (fs.existsSync(fa)) {
  for (const layer of ["best", "8"]) for (const k of [1, 10]) { check(fa, fb, "probes", layer, k); check(fb, fa, "all", layer, k); nPairs += 2; }
  const A = JSON.parse(fs.readFileSync(fa, "utf8")), B = JSON.parse(fs.readFileSync(fb, "utf8"));
  const rows = P.pairRows(A, B, "probes", "best", 10);
  const r1 = rows.find(r => r.a.id === "obj-01");
  const h = P.renderItem(r1, "A", "B", 10, false, A.meta, B.meta);
  eq("piece mark", /class="t hit piece">\s*cact/.test(h), true);
  eq("near mark", /class="t near">\s*desert/.test(h), true);
  eq("next label", /not next word/.test(h), true);
  eq("filter next word", P.filterRows(rows, { next: "next word" }).map(r => r.a.id), ["obj-02"]);
  eq("filter diff", P.filterRows(rows, { diff: "hard" }).map(r => r.a.id).sort(), ["emo-01", "garb-01"]);
  eq("esc", P.esc('<b a="x">&'), "&lt;b a=&quot;x&quot;&gt;&amp;");
} else console.log("no fake dumps (run scripts/qual/selftest.py first)");
// glue smoke test: run BOTH page scripts against a stub DOM + a fetch that reads site files; check the list renders
// and that a filter click and an item click re-render without errors.
async function smoke() {
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join("\n");
  const els = {}, handlers = {};
  const el = id => els[id] || (els[id] = { id, innerHTML: "", value: "", style: {}, dataset: {},
    parentNode: { style: {} }, addEventListener: (ev, f) => { handlers[id + ":" + ev] = f; } });
  el("k").value = "10";  // the static <select id="k"> (10 selected in the HTML)
  const store = {};
  const sandbox = { console, Promise, JSON, Math, Map, Set, Object, Array, String, Number, Error,
    document: { getElementById: el },
    localStorage: { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } },
    fetch: url => Promise.resolve({ ok: fs.existsSync(SITE + "/" + url), status: 404,
      json: () => Promise.resolve(JSON.parse(fs.readFileSync(SITE + "/" + url, "utf8"))) }) };
  vm.createContext(sandbox); vm.runInContext(scripts, sandbox);
  const tick = () => new Promise(r => setTimeout(r, 50));
  await tick(); await tick();
  const n0 = (el("list").innerHTML.match(/class="item"/g) || []).length;
  eq("smoke: items rendered", n0 > 50, true);
  eq("smoke: status", /hits A/.test(el("status").innerHTML), true);
  // set select value: opts() writes innerHTML and value; select's value must be a real option
  eq("smoke: lens A default J++", el("lensA").value, "9b/jpp.json");
  handlers["filters:click"]({ target: { closest: () => ({ dataset: { c: "improved" } }) } });
  await tick();
  const n1 = (el("list").innerHTML.match(/class="item"/g) || []).length;
  eq("smoke: improved filter", n1 > 0 && n1 < n0, true);
  const firstId = (el("list").innerHTML.match(/data-id="([^"]+)"/) || [])[1];
  handlers["list:click"]({ target: { closest: () => ({ dataset: { id: firstId.replace(/&amp;/g, "&") } }) } });
  await tick();
  eq("smoke: item opens all layers", /class="layers"/.test(el("list").innerHTML), true);
  eq("smoke: stored prefs", "lenscmp:model" in store, true);
}
smoke().then(() => {
  console.log(`${nPairs} pair configurations, ${nChecks} checks, ${fails} failures`);
  process.exit(fails ? 1 : 0);
}).catch(e => { console.log("SMOKE CRASH", e); process.exit(1); });
