const D = window.DADOS;   // vem de dados.js (gerado por src/painel.py)
const MODELOS = ["persistencia", "ridge", "hgb", "mlp", "gru"];
const NOME = { persistencia: "Persistência", ridge: "Ridge", hgb: "Gradient boosting", mlp: "MLP", gru: "GRU" };
const EST = Object.fromEntries(D.estacoes.map(s => [s.codigo, s]));
const ALVOS = D.estacoes.filter(s => s.papel === "alvo").map(s => s.codigo);
const EVENTOS = Object.keys(D.eventos_janelas);
const fmt = d3.format(",.0f"), fmt1 = d3.format(",.1f"), fmt2 = d3.format(".2f");
const ptBR = d3.formatLocale({ decimal: ",", thousands: ".", grouping: [3], currency: ["R$", ""] });
const n0 = ptBR.format(",.0f"), n1 = ptBR.format(",.1f"), n2 = ptBR.format(",.2f"), sgn0 = ptBR.format("+,.0f"), sgn1 = ptBR.format("+,.1f");
const utc = d3.utcParse("%Y-%m-%dT%H:%M"), utcD = d3.utcParse("%Y-%m-%d");
const ptTime = d3.timeFormatLocale({ dateTime: "%d/%m/%Y %H:%M", date: "%d/%m/%Y", time: "%H:%M", periods: ["", ""],
  days: ["domingo","segunda","terça","quarta","quinta","sexta","sábado"], shortDays: ["dom","seg","ter","qua","qui","sex","sáb"],
  months: ["janeiro","fevereiro","março","abril","maio","junho","julho","agosto","setembro","outubro","novembro","dezembro"],
  shortMonths: ["jan","fev","mar","abr","mai","jun","jul","ago","set","out","nov","dez"] });
const fDia = ptTime.utcFormat("%d/%m"), fHora = ptTime.utcFormat("%d/%m %Hh"), fData = ptTime.utcFormat("%d/%m/%Y"), fAno = ptTime.utcFormat("%Y");
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cor = m => css(`--m-${m}`);

const S = { est: "86510000", ev: "mai/2024", h: String(D.horizontes.at(-1)), modelos: new Set(MODELOS), livre: false, mapa: "mai/2024", bacia: "86720000", vh: String(D.horizontes[0]) };
try { const saved = JSON.parse(localStorage.getItem("taquari-painel") || "null"); if (saved) Object.assign(S, saved, { modelos: new Set(saved.modelos || MODELOS) }); } catch (e) {}
if (!D.horizontes.map(String).includes(S.h)) S.h = String(D.horizontes.at(-1));
if (!D.horizontes.map(String).includes(S.vh)) S.vh = String(D.horizontes[0]);
function persist() { try { localStorage.setItem("taquari-painel", JSON.stringify({ ...S, modelos: [...S.modelos] })); } catch (e) {} }

/* ---------------------------------------------------------------- tooltip */
const tip = document.getElementById("tip");
function showTip(ev, html) {
  tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect(), pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tip.hidden = true; };
const row = (label, val, c) => `<div class="r"><span>${c ? `<i style="--c:${c}"></i>` : ""}${label}</span><b>${val}</b></div>`;

/* ---------------------------------------------------------------- controles */
function seg(id, items, key, onchange) {
  const el = document.getElementById(id); el.innerHTML = "";
  items.forEach(([val, label]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.id = `${id}-${String(val).replace(/\W/g, "")}`;
    b.setAttribute("aria-pressed", String(S[key] === val));
    b.onclick = () => { S[key] = val; persist(); [...el.children].forEach(c => c.setAttribute("aria-pressed", String(c === b))); onchange(); };
    el.appendChild(b);
  });
}

/* ---------------------------------------------------------------- cabeçalho */
function renderFacts() {
  const t = D.teste.filter(r => r.modelo !== "persistencia");
  const best = h => d3.max(t.filter(r => r.horizonte_h === h), r => r.skill_vs_persist);
  const facts = [
    [n2(best(D.horizontes[0])), `melhor skill em ${D.horizontes[0]} h`],
    [n2(best(D.horizontes.at(-1))), `melhor skill em ${D.horizontes.at(-1)} h`],
    [n0(d3.max(D.eventos_stats.filter(r => r.evento === "mai/2024" && String(r.estacao) === "86510000"), r => r.pico_obs_cm)) , "cm · pico em Muçum, mai/2024"],
  ];
  document.getElementById("facts").innerHTML = facts.map(([v, l]) => `<div class="fact"><b>${v}</b><span>${l}</span></div>`).join("");
  document.getElementById("lede").textContent = `Previsão da cota em Muçum e Encantado com ${D.horizontes.slice(0, -1).join(", ")} e ${D.horizontes.at(-1)} horas de antecedência, avaliada nas cheias de 2023 e 2024.`;
  document.getElementById("foot-cenario").textContent =
    `Cenário: ${D.cenario === "com_imerg" ? "com chuva IMERG" : "sem IMERG"}; entradas e variação prevista limitadas à faixa do treino ± ${Math.round(100 * D.margem)}% da amplitude.`;
}

/* ---------------------------------------------------------------- mapa */
const K = Math.cos(29.2 * Math.PI / 180);
const tx = c => typeof c[0] === "number" ? [c[0] * K, c[1]] : c.map(tx);
const BACIAS = D.bacias.features.map(f => ({ ...f, geometry: { ...f.geometry, coordinates: tx(f.geometry.coordinates) } }))
  .sort((a, b) => b.properties.area_km2_poligono - a.properties.area_km2_poligono);
const G = D.chuva_grade;
const CELLS = [];
G.lat.forEach((la, i) => G.lon.forEach((lo, j) => {
  const k = i * G.lon.length + j, d = 0.05;
  CELLS.push({ k, lon: lo, lat: la, dentro: G.dentro[k] === 1, type: "Feature",
    geometry: { type: "Polygon", coordinates: [[[(lo - d) * K, la - d], [(lo + d) * K, la - d], [(lo + d) * K, la + d], [(lo - d) * K, la + d], [(lo - d) * K, la - d]]] } });
}));

function rampScale(vals) {
  const steps = d3.range(7).map(i => css(`--seq-${i}`));
  const v = vals.filter(x => x != null);
  const lo = d3.quantile(v, 0.02), hi = d3.quantile(v, 0.98);
  return { s: d3.scaleLinear().domain(d3.range(7).map(i => lo + (hi - lo) * i / 6)).range(steps).clamp(true), lo, hi, steps };
}

function renderMapa() {
  const svg = d3.select("#mapa"), box = svg.node().getBoundingClientRect();
  const W = Math.max(280, box.width), H = Math.round(W * 0.74);
  svg.attr("viewBox", `0 0 ${W} ${H}`).attr("height", H).selectAll("*").remove();
  const proj = d3.geoIdentity().reflectY(true).fitExtent([[4, 4], [W - 4, H - 4]], { type: "FeatureCollection", features: CELLS });
  const path = d3.geoPath(proj);
  const L = G.camadas[S.mapa], R = rampScale(L.v);

  svg.append("g").selectAll("path").data(CELLS).join("path").attr("d", path)
    .attr("fill", d => L.v[d.k] == null ? "transparent" : R.s(L.v[d.k]))
    .attr("fill-opacity", d => d.dentro ? 1 : 0.32)
    .on("mousemove", (ev, d) => showTip(ev, `<div class="t">${n2(d.lat)}°, ${n2(d.lon)}°</div>${row(L.rotulo, `${n0(L.v[d.k])} ${L.unidade}`)}${row("Na bacia de Encantado", d.dentro ? "sim" : "não")}`))
    .on("mouseleave", hideTip);

  const sel = S.bacia;
  const bacias = [...BACIAS].sort((a, b) => (a.properties.codigo === sel) - (b.properties.codigo === sel));
  svg.append("g").selectAll("path").data(bacias).join("path").attr("d", path)
    .attr("fill", css("--silt")).attr("fill-opacity", d => d.properties.codigo === sel ? 0.16 : 0)
    .attr("stroke", d => d.properties.codigo === sel ? css("--silt") : css("--ink"))
    .attr("stroke-width", d => d.properties.codigo === sel ? 2.5 : (EST[d.properties.codigo].papel === "alvo" ? 1.4 : 0.8))
    .attr("stroke-dasharray", d => EST[d.properties.codigo].papel === "alvo" || d.properties.codigo === sel ? null : "3 3")
    .attr("stroke-linejoin", "round").style("cursor", "pointer")
    .on("mousemove", (ev, d) => { const p = d.properties;
      showTip(ev, `<div class="t">${p.codigo}</div><div style="font-weight:600;margin-bottom:4px">${p.nome}</div>${row("Área (polígono)", `${n0(p.area_km2_poligono)} km²`)}${row("Área (ANA)", `${n0(p.area_km2_ana)} km²`)}${row("Diferença", `${sgn1(p["dif_area_%"])}%`)}`); })
    .on("mouseleave", hideTip)
    .on("click", (ev, d) => { S.bacia = d.properties.codigo; persist(); renderMapa(); renderStations(); });

  const st = svg.append("g");
  D.estacoes.forEach(s => {
    const [x, y] = proj([s.lon * K, s.lat]);
    const g = st.append("g").attr("transform", `translate(${x},${y})`);
    g.append("path").attr("d", "M0,-7 L6,4 L-6,4 Z").attr("fill", s.papel === "alvo" ? css("--ink") : css("--surface")).attr("stroke", s.papel === "alvo" ? css("--surface") : css("--ink")).attr("stroke-width", 1.5);
    const left = s.codigo === "86510000" || s.codigo === "86720000";
    g.append("text").attr("x", left ? -10 : 10).attr("y", s.codigo === "86720000" ? 14 : -4).attr("text-anchor", left ? "end" : "start")
      .style("fill", css("--ink")).style("font-family", "var(--f-body)").style("font-size", "12px").style("font-weight", 600)
      .style("paint-order", "stroke").style("stroke", css("--surface")).style("stroke-width", 3).text(s.nome);
  });

  const km = 50, px = km / 111.32 * proj.scale();
  const sb = svg.append("g").attr("transform", `translate(${W - px - 18},${H - 18})`);
  sb.append("rect").attr("x", -6).attr("y", -16).attr("width", px + 12).attr("height", 26).attr("rx", 3).attr("fill", css("--surface")).attr("fill-opacity", .85);
  sb.append("rect").attr("width", px).attr("height", 4).attr("fill", css("--ink"));
  sb.append("text").attr("x", px / 2).attr("y", -6).attr("text-anchor", "middle").style("fill", css("--ink-2")).text(`${km} km`);

  const lg = document.getElementById("leg-mapa");
  lg.innerHTML = `<span class="eyebrow">${L.rotulo} (${L.unidade})</span><div class="bar" style="background:linear-gradient(90deg,${R.steps.join(",")})"></div><div class="ticks"><span>≤ ${n0(R.lo)}</span><span>${n0((R.lo + R.hi) / 2)}</span><span>≥ ${n0(R.hi)}</span></div>`;
  renderRainSum();
}

function renderStations() {
  const el = document.getElementById("stations");
  el.innerHTML = "";
  [...D.bacias.features].sort((a, b) => a.properties.area_km2_poligono - b.properties.area_km2_poligono).forEach(f => {
    const p = f.properties, s = EST[p.codigo];
    const b = document.createElement("button"); b.type = "button"; b.className = "st-row"; b.id = `st-${p.codigo}`;
    b.setAttribute("aria-pressed", String(S.bacia === p.codigo));
    b.innerHTML = `<div><div class="nm">${p.nome}<span class="tag ${s.papel === "alvo" ? "alvo" : ""}">${s.papel === "alvo" ? "alvo" : "montante"}</span></div><div class="cd">${p.codigo} · ${n2(s.lat)}°, ${n2(s.lon)}°</div></div>
      <div class="ar">${n0(p.area_km2_poligono)} km²<small>ANA ${n0(p.area_km2_ana)} · ${sgn1(p["dif_area_%"])}%</small></div>`;
    b.onclick = () => { S.bacia = p.codigo; persist(); renderMapa(); renderStations(); };
    el.appendChild(b);
  });
}

function renderRainSum() {
  const L = G.camadas[S.mapa];
  const ins = CELLS.filter(c => c.dentro && L.v[c.k] != null);
  const w = ins.map(c => Math.cos(c.lat * Math.PI / 180));
  const mean = d3.sum(ins, (c, i) => L.v[c.k] * w[i]) / d3.sum(w);
  const out = CELLS.filter(c => !c.dentro && L.v[c.k] != null);
  const meanOut = d3.mean(out, c => L.v[c.k]);
  document.getElementById("rain-sum").innerHTML =
    `<span class="eyebrow">${L.rotulo}</span><b>${n0(mean)} ${L.unidade}</b>
     <span class="note">Média nas ${ins.length} células da bacia de Encantado. Fora dela, na mesma caixa: ${n0(meanOut)} ${L.unidade}.</span>`;
}

/* ---------------------------------------------------------------- evento */
function eventData() {
  const E = D.eventos[`${S.est}|${S.ev}`], H = E.horizontes[S.h];
  const t0 = utc(E.inicio), T = d3.range(E.n).map(i => new Date(+t0 + i * 36e5));
  return { E, H, T };
}

function gapRuns(obs, T) {
  const runs = []; let s = null;
  const first = obs.findIndex(v => v != null), last = obs.length - 1 - [...obs].reverse().findIndex(v => v != null);
  for (let i = 0; i < obs.length; i++) {
    const miss = obs[i] == null;
    if (miss && s == null) s = i;
    if ((!miss || i === obs.length - 1) && s != null) { const e = miss ? i : i - 1; if (e - s + 1 >= 3) runs.push([s, e]); s = null; }
  }
  return { runs, first, last };
}

function renderEvento() {
  const { E, H, T } = eventData();
  const box = document.getElementById("ev-chart").getBoundingClientRect();
  const W = Math.max(300, box.width), Hh = W < 520 ? 280 : 360, m = { t: 26, r: 14, b: 26, l: 52 };
  const svg = d3.select("#ev-chart").attr("viewBox", `0 0 ${W} ${Hh}`).attr("height", Hh);
  svg.selectAll("*").remove();
  const ativos = MODELOS.filter(mo => S.modelos.has(mo));
  const series = [];
  ativos.forEach(mo => {
    series.push({ id: mo, v: H.modelos[mo], c: cor(mo), dash: mo === "persistencia" ? "5 4" : null, w: mo === "persistencia" ? 1.5 : 2, nome: NOME[mo] });
    if (S.livre && H.modelos[mo + "__livre"]) series.push({ id: mo + "__livre", v: H.modelos[mo + "__livre"], c: cor(mo), dash: "2 3", w: 1.5, nome: `${NOME[mo]} sem limites` });
  });
  const allv = H.obs.concat(...series.map(s => s.v)).filter(v => v != null);
  const x = d3.scaleUtc().domain(d3.extent(T)).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allv)), d3.max(allv) * 1.05]).nice().range([Hh - m.b, m.t]);

  svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(W - m.l - m.r)).tickFormat(""));
  const [a, b] = E.janela;
  svg.append("rect").attr("x", x(utcD(a))).attr("width", x(new Date(+utcD(b) + 864e5)) - x(utcD(a))).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", css("--band-ev"));
  const { runs } = gapRuns(H.obs, T);
  const gg = svg.append("g");
  runs.forEach(([s, e]) => {
    const x0 = x(T[s]), x1 = x(T[e]) + (x(T[1]) - x(T[0]));
    gg.append("rect").attr("x", x0).attr("width", Math.max(1, x1 - x0)).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", css("--gap"));
    if (x1 - x0 > 60) gg.append("text").attr("x", (x0 + x1) / 2).attr("y", m.t + 14).attr("text-anchor", "middle").style("fill", css("--ink-2")).text("sem medição");
  });

  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).ticks(W < 520 ? 5 : 9).tickFormat(fDia));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(n0)).call(g => g.select("path").remove());
  svg.append("text").attr("x", 4).attr("y", 12).text("cota (cm)");

  const line = d3.line().defined(d => d[1] != null).x(d => x(d[0])).y(d => y(d[1]));
  const pts = v => T.map((t, i) => [t, v[i]]);
  series.forEach(s => svg.append("path").attr("d", line(pts(s.v))).attr("fill", "none").attr("stroke", s.c).attr("stroke-width", s.w).attr("stroke-dasharray", s.dash).attr("stroke-linejoin", "round"));
  svg.append("path").attr("d", line(pts(H.obs))).attr("fill", "none").attr("stroke", css("--surface")).attr("stroke-width", 5.5);
  svg.append("path").attr("d", line(pts(H.obs))).attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 2.5).attr("stroke-linejoin", "round");

  const io = d3.maxIndex(H.obs.map(v => v ?? -Infinity));
  if (H.obs[io] != null) {
    svg.append("circle").attr("cx", x(T[io])).attr("cy", y(H.obs[io])).attr("r", 4.5).attr("fill", css("--ink")).attr("stroke", css("--surface")).attr("stroke-width", 2);
    svg.append("text").attr("x", x(T[io]) + 8).attr("y", y(H.obs[io]) - 6).style("fill", css("--ink")).style("font-weight", 600).text(`observado ${n0(H.obs[io])}`);
  }

  const cross = svg.append("line").attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", css("--ink-3")).attr("stroke-width", 1).style("display", "none");
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", Hh - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", ev => {
      const [px] = d3.pointer(ev), i = Math.max(0, Math.min(T.length - 1, Math.round((x.invert(px) - T[0]) / 36e5)));
      cross.attr("x1", x(T[i])).attr("x2", x(T[i])).style("display", null);
      const v = val => val == null ? "—" : `${n0(val)} cm`;
      showTip(ev, `<div class="t">${fHora(T[i])} (hora local)</div>${row("Observado", v(H.obs[i]), css("--ink"))}${series.map(s => row(s.nome, v(s.v[i]), s.c)).join("")}`);
    })
    .on("mouseleave", () => { cross.style("display", "none"); hideTip(); });

  renderChuvaEvento(E, x, W, m);
  renderRegua(H, series, y, Hh, m);
  renderEvTable();
}

function renderChuvaEvento(E, x, W, m) {
  const svg = d3.select("#ev-rain"), Hh = 92, mm = { t: 6, b: 20 };
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).attr("height", Hh).selectAll("*").remove();
  const passo = (E.chuva.passo_h || 24) * 36e5, horaria = passo < 864e5;
  const d0 = horaria ? utc(E.chuva.inicio) : utcD(E.chuva.inicio), days = E.chuva.v.map((v, i) => ({ t: new Date(+d0 + i * passo), v }));
  const y = d3.scaleLinear().domain([0, Math.max(horaria ? 4 : 20, d3.max(days, d => d.v) || 0)]).nice().range([Hh - mm.b, mm.t]);
  const [x0, x1] = x.domain();
  svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSize(-(W - m.l - m.r)).tickFormat(""));
  const bw = x(new Date(+x0 + passo)) - x(x0), gap = horaria ? 0 : 1;
  svg.append("g").selectAll("path").data(days.filter(d => d.v != null && d.t >= new Date(+x0 - passo) && d.t <= x1)).join("path")
    .attr("d", d => { const xl = Math.max(m.l, x(d.t) + gap), xr = Math.min(W - m.r, x(d.t) + bw - gap), yt = y(d.v), yb = y(0), r = Math.min(horaria ? 1 : 4, (xr - xl) / 2, yb - yt);
      if (xr <= xl || yb - yt < .5) return ""; return `M${xl},${yb}V${yt + r}Q${xl},${yt} ${xl + r},${yt}H${xr - r}Q${xr},${yt} ${xr},${yt + r}V${yb}Z`; })
    .attr("fill", css("--seq-3"))
    .on("mousemove", (ev, d) => showTip(ev, horaria ? `<div class="t">${fHora(d.t)} (hora local)</div>${row("Chuva na bacia (IMERG Early)", `${n1(d.v)} mm/h`, css("--seq-3"))}` : `<div class="t">${fData(d.t)} (dia UTC)</div>${row("Chuva na bacia", `${n1(d.v)} mm`, css("--seq-3"))}`))
    .on("mouseleave", hideTip);
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickFormat(n0)).call(g => g.select("path").remove());
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - mm.b})`).call(d3.axisBottom(x).ticks(W < 520 ? 5 : 9).tickFormat(fDia));
}

function renderRegua(H, series, yMain, Hh, m) {
  const svg = d3.select("#regua"), W = 174;
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).attr("width", "100%").attr("height", Hh).style("max-width", W + "px").selectAll("*").remove();
  const y = yMain.copy();
  const [lo, hi] = y.domain(), sx = 14, sw = 16;
  for (let c = Math.ceil(lo / 100) * 100; c < hi; c += 100) {
    const y0 = y(c), y1 = y(Math.min(hi, c + 100));
    svg.append("rect").attr("x", sx).attr("y", y1).attr("width", sw).attr("height", y0 - y1)
      .attr("fill", (c / 100) % 2 === 0 ? css("--ink") : css("--surface")).attr("stroke", css("--ink")).attr("stroke-width", .6);
    if (c % 500 === 0) svg.append("text").attr("x", sx - 3).attr("y", y0 + 3).attr("text-anchor", "end").style("font-size", "9px").text(c / 100);
  }
  svg.append("text").attr("x", sx - 3).attr("y", m.t - 3).attr("text-anchor", "start").style("font-size", "9px").text("m");
  const marks = [];
  const io = d3.maxIndex(H.obs.map(v => v ?? -Infinity));
  if (H.obs[io] != null) marks.push({ v: H.obs[io], c: css("--ink"), l: "obs", strong: true });
  series.filter(s => !s.id.endsWith("__livre")).forEach(s => { const v = d3.max(s.v); if (v != null) marks.push({ v, c: s.c, l: NOME[s.id] === "Gradient boosting" ? "HGB" : NOME[s.id].replace("Persistência", "Persist.") }); });
  marks.sort((a, b) => b.v - a.v);
  let lastY = -Infinity; marks.forEach(mk => { mk.ty = Math.max(y(mk.v), lastY + 14); lastY = mk.ty; });
  const over = lastY - (Hh - m.b); if (over > 0) marks.forEach(mk => { mk.ty -= over; });
  marks.forEach(mk => {
    const yy = y(mk.v);
    svg.append("line").attr("x1", sx + sw).attr("x2", sx + sw + 14).attr("y1", yy).attr("y2", yy).attr("stroke", mk.c).attr("stroke-width", mk.strong ? 2.5 : 2);
    svg.append("path").attr("d", `M${sx + sw + 14},${yy} L${sx + sw + 22},${mk.ty}`).attr("stroke", css("--rule")).attr("fill", "none");
    svg.append("text").attr("x", sx + sw + 26).attr("y", mk.ty + 4).style("fill", mk.strong ? css("--ink") : css("--ink-2")).style("font-weight", mk.strong ? 600 : 400)
      .text(`${mk.l} ${n0(mk.v)}`);
  });
}

function renderEvTable() {
  const rows = D.eventos_stats.filter(r => String(r.estacao) === S.est && r.evento === S.ev && String(r.horizonte_h) === S.h);
  const ef = D.efeito.filter(r => String(r.estacao) === S.est && r.evento === S.ev && String(r.horizonte_h) === S.h);
  const byM = Object.fromEntries(rows.map(r => [r.modelo, r]));
  const efM = Object.fromEntries(ef.map(r => [r.modelo, r]));
  const t = document.getElementById("ev-table");
  t.innerHTML = `<thead><tr><th>Modelo</th><th>Pico previsto<br>(cm)</th><th>Erro de pico<br>(cm)</th><th>Erro de pico<br>(%)</th><th>Sem limites:<br>erro de pico (cm)</th><th>Erro de horário<br>(h)</th><th>RMSE no evento<br>(cm)</th></tr></thead><tbody>${
    MODELOS.filter(mo => byM[mo]).map(mo => { const r = byM[mo], e = efM[mo];
      const livre = e && e.erro_pico_cm_sem != null ? sgn0(e.erro_pico_cm_sem) : "—";
      return `<tr><td><span class="sw" style="background:${cor(mo)}"></span>${NOME[mo]}</td><td>${n0(r.pico_prev_cm)}</td><td class="pos">${sgn0(r.erro_pico_cm)}</td><td>${sgn1(r["erro_pico_%"])}</td><td class="dim">${livre}</td><td>${sgn0(r.erro_horario_h)}</td><td>${n0(r.RMSE_evento_cm)}</td></tr>`; }).join("")
  }</tbody>`;
  const r0 = rows[0], al = document.getElementById("ev-alert");
  if (r0 && (r0.pico_perto_de_lacuna || r0["cobertura_%"] < 90)) {
    al.hidden = false;
    document.getElementById("ev-alert-txt").innerHTML = `<b>Pico não medido.</b> O sensor de ${EST[S.est].nome} ficou sem dados perto do pico (cobertura de ${n0(r0["cobertura_%"])}% da janela). O "pico observado" de ${n0(r0.pico_obs_cm)} cm é só o último valor registrado, então o erro de pico desta combinação não é confiável.`;
  } else al.hidden = true;
}

function renderChips() {
  const el = document.getElementById("chips"); el.innerHTML = "";
  MODELOS.forEach(mo => {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip" + (mo === "persistencia" ? " dash" : ""); b.id = `chip-${mo}`;
    b.style.setProperty("--c", cor(mo)); b.setAttribute("aria-pressed", String(S.modelos.has(mo)));
    b.innerHTML = `<span class="sw"></span>${NOME[mo]}`;
    b.onclick = () => { S.modelos.has(mo) ? S.modelos.delete(mo) : S.modelos.add(mo); persist(); renderChips(); renderEvento(); };
    el.appendChild(b);
  });
}

/* ---------------------------------------------------------------- skill */
function renderSkill() {
  const host = document.getElementById("skill"); host.innerHTML = "";
  const mods = MODELOS.filter(m => m !== "persistencia");
  document.getElementById("leg-skill").innerHTML = mods.map(m => `<span><i style="--c:${cor(m)}"></i>${NOME[m]}</span>`).join("");
  const all = D.teste.filter(r => r.modelo !== "persistencia").map(r => r.skill_vs_persist);
  ALVOS.forEach(cod => {
    const box = document.createElement("div"); box.className = "panel small-box";
    box.innerHTML = `<h3>${EST[cod].nome}</h3>`; host.appendChild(box);
    const W = Math.max(280, box.getBoundingClientRect().width - 24), Hh = 250, m = { t: 12, r: 70, b: 28, l: 40 };
    const svg = d3.select(box).append("svg").attr("viewBox", `0 0 ${W} ${Hh}`).attr("height", Hh).attr("width", "100%");
    const x = d3.scalePoint().domain(D.horizontes).range([m.l + 20, W - m.r]);
    const y = d3.scaleLinear().domain([Math.min(-0.2, d3.min(all)), 1]).nice().range([Hh - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r + 50)).tickFormat(""));
    svg.append("line").attr("x1", m.l).attr("x2", W - 8).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--ink-3")).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.5);
    svg.append("text").attr("x", W - 8).attr("y", y(0) - 5).attr("text-anchor", "end").text("persistência");
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(n1)).call(g => g.select("path").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).tickFormat(h => `${h} h`)).call(g => g.select("path").remove());
    mods.forEach(mo => {
      const rs = D.horizontes.map(h => D.teste.find(r => String(r.estacao) === cod && r.modelo === mo && r.horizonte_h === h)).filter(Boolean);
      svg.append("path").attr("d", d3.line().x(r => x(r.horizonte_h)).y(r => y(r.skill_vs_persist))(rs)).attr("fill", "none").attr("stroke", cor(mo)).attr("stroke-width", 2);
      svg.append("g").selectAll("circle").data(rs).join("circle").attr("cx", r => x(r.horizonte_h)).attr("cy", r => y(r.skill_vs_persist)).attr("r", 4.5)
        .attr("fill", cor(mo)).attr("stroke", css("--surface")).attr("stroke-width", 2);
      svg.append("g").selectAll("circle").data(rs).join("circle").attr("cx", r => x(r.horizonte_h)).attr("cy", r => y(r.skill_vs_persist)).attr("r", 12).attr("fill", "transparent")
        .on("mousemove", (ev, r) => { const p = D.teste.find(q => String(q.estacao) === cod && q.modelo === "persistencia" && q.horizonte_h === r.horizonte_h);
          showTip(ev, `<div class="t">${EST[cod].nome} · ${r.horizonte_h} h</div>${row(NOME[mo], `skill ${n2(r.skill_vs_persist)}`, cor(mo))}${row("RMSE do modelo", `${n1(r.RMSE_cm)} cm`)}${row("RMSE da persistência", `${n1(p.RMSE_cm)} cm`)}${row("NSE", n2(r.NSE))}`); })
        .on("mouseleave", hideTip);
    });
    const best = mods.map(mo => D.teste.find(r => String(r.estacao) === cod && r.modelo === mo && r.horizonte_h === D.horizontes.at(-1))).sort((a, b) => b.skill_vs_persist - a.skill_vs_persist);
    const lab = [];
    best.forEach(r => { let ty = y(r.skill_vs_persist) + 4; lab.forEach(l => { if (Math.abs(l - ty) < 13) ty = l + 13; }); lab.push(ty);
      svg.append("text").attr("x", x(D.horizontes.at(-1)) + 9).attr("y", ty).style("fill", css("--ink-2")).text(NOME[r.modelo] === "Gradient boosting" ? "HGB" : NOME[r.modelo]); });
  });
}

/* ---------------------------------------------------------------- série longa */
function renderLong() {
  const host = document.getElementById("long"); host.innerHTML = "";
  const L = D.serie_longa, d0 = utcD(L.inicio);
  ALVOS.forEach(cod => {
    const v = L.series[cod], T = v.map((_, i) => new Date(+d0 + i * 864e5));
    const box = document.createElement("div"); box.className = "panel small-box";
    box.innerHTML = `<h3>${EST[cod].nome} <span class="num muted" style="font-size:.8rem;font-family:var(--f-mono)">${cod}</span></h3>`; host.appendChild(box);
    const W = Math.max(280, box.getBoundingClientRect().width - 24), Hh = 230, m = { t: 22, r: 10, b: 26, l: 48 };
    const svg = d3.select(box).append("svg").attr("viewBox", `0 0 ${W} ${Hh}`).attr("height", Hh).attr("width", "100%");
    const x = d3.scaleUtc().domain(d3.extent(T)).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(D.serie_longa.series[ALVOS[0]].concat(D.serie_longa.series[ALVOS[1]]))]).nice().range([Hh - m.b, m.t]);
    const band = (a, b, f, label) => { const xa = x(Math.max(+T[0], +utcD(a))), xb = x(Math.min(+T[T.length - 1], +utcD(b)));
      svg.append("rect").attr("x", xa).attr("width", Math.max(0, xb - xa)).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", f);
      if (label && xb - xa > 34) svg.append("text").attr("x", (xa + xb) / 2).attr("y", m.t - 7).attr("text-anchor", "middle").text(label); };
    band(L.inicio, L.treino_fim, css("--band-train"), "treino");
    band(L.treino_fim, L.val_fim, css("--band-val"), "valid.");
    band(L.teste_ini, fData(T[T.length - 1]).split("/").reverse().join("-"), "transparent", "teste");
    Object.entries(D.eventos_janelas).forEach(([n, [a, b]]) => band(a, b, css("--band-ev")));
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("path").attr("d", d3.line().defined(d => d[1] != null).x(d => x(d[0])).y(d => y(d[1]))(T.map((t, i) => [t, v[i]])))
      .attr("fill", "none").attr("stroke", css("--agua")).attr("stroke-width", 1.2);
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(n0)).call(g => g.select("path").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).ticks(d3.utcYear.every(W < 420 ? 2 : 1)).tickFormat(fAno));
    if (cod === "86720000") {
      const xs = x(utcD("2024-07-01")), ys = y(450);
      svg.append("path").attr("d", `M${xs},${ys} L${xs + 14},${ys - 38}`).attr("stroke", css("--ink-3")).attr("fill", "none");
      svg.append("text").attr("x", xs + 16).attr("y", ys - 42).attr("text-anchor", W - xs < 140 ? "end" : "start").style("fill", css("--ink-2"))
        .text("base cai ~1,5 m depois de 2024");
    }
    const cross = svg.append("line").attr("y1", m.t).attr("y2", Hh - m.b).attr("stroke", css("--ink-3")).style("display", "none");
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", Hh - m.t - m.b).attr("fill", "transparent")
      .on("mousemove", ev => { const [px] = d3.pointer(ev), i = Math.max(0, Math.min(T.length - 1, Math.round((x.invert(px) - T[0]) / 864e5)));
        cross.attr("x1", x(T[i])).attr("x2", x(T[i])).style("display", null);
        showTip(ev, `<div class="t">${fData(T[i])}</div>${row("Cota máxima do dia", v[i] == null ? "sem dado" : `${n0(v[i])} cm`, css("--ink-2"))}`); })
      .on("mouseleave", () => { cross.style("display", "none"); hideTip(); });
  });
}

/* ---------------------------------------------------------------- agora (ao vivo) */
let LIVE = null, LIVE_ERR = null;
const nowLocal = () => new Date(Date.now() + (D.ao_vivo?.utc_offset_h ?? -3) * 36e5);   // relógio local no mesmo "UTC falso" das séries
const REF_NOMES = [["atencao", "atenção"], ["alerta", "alerta"], ["inundacao", "inundação"]];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function loadLive() {
  const urls = (D.ao_vivo?.urls || []).concat(["agora.json"]);
  for (const u of urls) {
    try {
      const r = await fetch(u + (u.includes("?") ? "&" : "?") + "t=" + Math.floor(Date.now() / 3e5), { cache: "no-store" });
      if (!r.ok) continue;
      const js = await r.json();
      if (js && js.estacoes) { LIVE = js; LIVE_ERR = null; renderAgora(); irParaAncora(); return; }
    } catch (e) { LIVE_ERR = e; }
  }
  LIVE = null; renderAgora(); irParaAncora();
}

/* A seção "Agora" muda de altura quando os dados ao vivo chegam; um link direto (#cheias etc.) precisa ser refeito uma vez. */
let ancoraFeita = false;
function irParaAncora() {
  if (ancoraFeita) return;
  ancoraFeita = true;
  const el = location.hash && document.querySelector(location.hash);
  if (!el) return;
  // fontes e gráficos ainda mudam a altura da página por um instante: reposiciona algumas vezes,
  // e para assim que a pessoa rolar por conta própria
  let mexeu = false;
  const parar = () => { mexeu = true; };
  ["wheel", "touchstart", "keydown", "mousedown"].forEach(t => addEventListener(t, parar, { once: true, passive: true }));
  const vai = () => { if (!mexeu) el.scrollIntoView({ block: "start", behavior: "instant" }); };
  vai();
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(vai);
  [300, 800, 1600].forEach(ms => setTimeout(vai, ms));
}

function liveStatus() {
  const el = document.getElementById("live-status"), nav = document.getElementById("nav-live");
  if (!LIVE) {
    el.innerHTML = `<span class="live-dot"></span>sem conexão com os dados ao vivo`;
    nav.innerHTML = `<span class="live-dot"></span><span class="txt">ao vivo indisponível</span>`;
    return;
  }
  const g = utc(LIVE.gerado_em), min = Math.round((nowLocal() - g) / 6e4);
  const quando = min < 60 ? `há ${Math.max(0, min)} min` : min < 48 * 60 ? `há ${Math.round(min / 60)} h` : `em ${fData(g)}`;
  const estado = min <= 150 ? "on" : "old";
  el.innerHTML = `<span class="live-dot ${estado}"></span>atualizado ${quando} · ${fHora(g)} (hora local)`;
  nav.innerHTML = `<span class="live-dot ${estado}"></span><span class="txt">ao vivo · ${quando}</span>`;
  nav.title = `Dados ao vivo atualizados ${quando} (${fHora(g)}, hora local)`;
}

function nivelRef(v, ref) {
  if (!ref || v == null) return null;
  if (v >= ref.inundacao) return "acima da cota de inundação";
  if (v >= ref.alerta) return "acima da cota de alerta";
  if (v >= ref.atencao) return "acima da cota de atenção";
  return "abaixo da cota de atenção";
}

function renderAgora() {
  liveStatus();
  renderConferencia();
  const body = document.getElementById("live-body"), av = document.getElementById("live-avisos");
  if (!LIVE) {
    av.hidden = true;
    body.innerHTML = `<div class="panel live-off">Os dados ao vivo não puderam ser carregados agora. O restante da página (avaliação histórica) continua disponível abaixo.</div>`;
    return;
  }
  const L = LIVE, avisos = [...(L.avisos || [])];
  const g = utc(L.gerado_em);
  if ((nowLocal() - g) / 36e5 > 3) avisos.unshift(`Os dados têm mais de 3 horas (última atualização ${fHora(g)}). A rotina automática pode estar atrasada.`);
  av.hidden = !avisos.length;
  document.getElementById("live-avisos-txt").innerHTML = avisos.map(esc).join("<br>");

  const T0 = utc(L.inicio), T = d3.range(L.n).map(i => new Date(+T0 + i * 36e5));
  const alvos = Object.entries(L.estacoes).filter(([, e]) => e.papel === "alvo");
  const montante = Object.entries(L.estacoes).filter(([, e]) => e.papel !== "alvo");
  body.innerHTML = `<div class="two">${alvos.map(([c]) => `<div class="panel now-card" id="now-${c}"></div>`).join("")}</div>
    <div class="now-sub" style="margin-top:16px">
      <div class="panel"><div class="eyebrow" style="padding:12px 14px 0">Estações a montante · últimas 72 h</div><div id="now-up"></div></div>
      <div class="panel small-box"><div class="chart-cap">Chuva média na bacia de Encantado: satélite IMERG Early (observada) e previsão do GFS (mm/h, hora local)</div><svg id="now-rain" role="img" aria-label="Chuva horária observada nas últimas 72 horas e prevista para as próximas 24 horas"></svg><div class="acc" id="now-acc"></div></div>
    </div>
    <div class="now-sub" style="margin-top:16px" id="now-pluvio-wrap">
      <div class="panel small-box"><div class="chart-cap">Pluviômetros na bacia (ANA, SGB, Defesa Civil RS) · chuva nas últimas 24 h</div><svg id="now-pluvio-map" role="img" aria-label="Mapa dos pluviômetros da bacia com a chuva das últimas 24 horas"></svg></div>
      <div class="panel small-box"><div class="chart-cap">Chuva observada em solo, com ~1 h de atraso (cobre as horas que o satélite ainda não mostra)</div><div class="acc" id="now-pluvio-acc"></div><div id="now-pluvio-list"></div></div></div>
    </div>`;
  alvos.forEach(([c, e]) => renderNowCard(c, e, T));
  document.getElementById("now-up").innerHTML = montante.map(([c, e]) => {
    const obs = e.obs || [], last = d3.findLastIndex ? d3.findLastIndex(obs, v => v != null) : obs.map(v => v != null).lastIndexOf(true);
    const v = last >= 0 ? obs[last] : null, v6 = last >= 6 ? obs[last - 6] : null;
    const tend = v != null && v6 != null ? `${sgn0(v - v6)} cm em 6 h` : "sem leitura recente";
    return `<div class="up-row"><div><div class="nm">${esc(e.nome)}</div><div class="note">${c}</div></div>
      <div class="v">${v == null ? "—" : n0(v) + " cm"}<small>${tend}</small></div><svg id="spark-${c}" aria-hidden="true"></svg></div>`;
  }).join("");
  montante.forEach(([c, e]) => spark(`#spark-${c}`, e.obs || [], T));
  renderNowRain(T);
  renderPluvio();
}

function renderPluvio() {
  const P = LIVE.pluviometros, wrap = document.getElementById("now-pluvio-wrap");
  if (!P) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const svg = d3.select("#now-pluvio-map"), box = svg.node().parentNode.getBoundingClientRect();
  const W = Math.max(260, box.width - 24), H = Math.round(W * 0.72);
  svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("height", H).selectAll("*").remove();
  const bacia = BACIAS.find(b => b.properties.codigo === "86720000");
  const proj = d3.geoIdentity().reflectY(true).fitExtent([[10, 10], [W - 10, H - 10]], bacia);
  svg.append("path").attr("d", d3.geoPath(proj)(bacia)).attr("fill", css("--agua")).attr("fill-opacity", .06)
    .attr("stroke", css("--agua")).attr("stroke-width", 1.6).attr("stroke-linejoin", "round");
  const est = P.estacoes.slice().sort((a, b) => (a.chuva_24h ?? -1) - (b.chuva_24h ?? -1));
  const vmax = Math.max(20, d3.max(est, e => e.chuva_24h) || 0);
  const r = d3.scaleSqrt().domain([0, vmax]).range([3, Math.max(10, W / 22)]);
  const cor = d3.scaleQuantize().domain([0, Math.max(40, vmax)]).range([1, 2, 3, 4, 5, 6].map(i => css(`--seq-${i}`)));
  svg.append("g").selectAll("circle").data(est).join("circle")
    .attr("cx", e => proj([e.lon * K, e.lat])[0]).attr("cy", e => proj([e.lon * K, e.lat])[1])
    .attr("r", e => e.chuva_24h == null ? 3 : r(e.chuva_24h))
    .attr("fill", e => e.chuva_24h == null ? "transparent" : cor(e.chuva_24h)).attr("fill-opacity", .9)
    .attr("stroke", e => e.chuva_24h == null ? css("--ink-3") : css("--surface")).attr("stroke-width", 1.2)
    .on("mousemove", (ev, e) => showTip(ev, `<div class="t">${esc(e.operadora)} · ${e.codigo}</div><div style="font-weight:600;margin-bottom:4px">${esc(e.nome)}</div>${row("Últimas 6 h", e.chuva_6h == null ? "—" : `${n1(e.chuva_6h)} mm`)}${row("Últimas 24 h", e.chuva_24h == null ? "sem dado" : `${n1(e.chuva_24h)} mm`)}${e.ultima ? row("Última leitura", fHora(utc(e.ultima))) : ""}`))
    .on("mouseleave", hideTip);
  const nomes = { bacia: "bacia inteira", antas: "Antas", taquari_sup: "Taquari sup.", guapore: "Guaporé" };
  const f = v => v == null ? "—" : `${n1(v)} mm`;
  document.getElementById("now-pluvio-acc").innerHTML = Object.keys(P.acumulado_24h).map(k =>
    `<div><b>${f(P.acumulado_24h[k])}</b><span>${nomes[k] || k} · 24 h (6 h: ${f(P.acumulado_6h[k])})</span></div>`).join("")
    + `<div><span>${P.n_estacoes} de ${P.n_rede} estações com dados · até ${fHora(utc(P.ultima_hora))} · média ponderada por polígonos de Thiessen</span></div>`;
  const top = P.estacoes.filter(e => e.chuva_24h != null).sort((a, b) => b.chuva_24h - a.chuva_24h).slice(0, 5);
  document.getElementById("now-pluvio-list").innerHTML = top.map(e => `<div class="up-row"><div><div class="nm">${esc(e.nome)}</div><div class="note">${esc(e.operadora)} · ${e.codigo}</div></div><div class="v">${n1(e.chuva_24h)} mm<small>6 h: ${f(e.chuva_6h)}</small></div><span></span></div>`).join("");
}

function renderNowCard(c, e, T) {
  const el = document.getElementById(`now-${c}`);
  const ref = e.cotas_referencia, fcs = e.previsoes || [];
  const tr = e.t_ref ? utc(e.t_ref) : null;
  const status = nivelRef(e.cota_t, ref);
  const maxPrev = fcs.length ? d3.max(fcs, q => q.cota) : null;
  const statusPrev = nivelRef(maxPrev, ref);
  el.innerHTML = `<div class="now-top">
      <div><h3>${esc(e.nome)} <span class="tag alvo">alvo</span></h3>
        <div class="now-level"><b>${e.cota_t == null ? "—" : n0(e.cota_t)}</b><span>cm${tr ? ` às ${fHora(tr)}` : ""}</span></div></div>
      <div class="now-fc">${fcs.map(q => `<div class="fc"><span class="h">+${q.h} h</span><b>${n0(q.cota)}</b><small>${sgn0(q.delta)} cm · ${ptTime.utcFormat("%Hh")(utc(q.t))}</small></div>`).join("")}</div>
    </div>
    ${e.erro ? `<div class="ref-line">Sem previsão: ${esc(e.erro)}.</div>` : ""}
    ${ref ? `<div class="ref-line">Cotas do SGB/SACE: atenção <b>${n0(ref.atencao)}</b>, alerta <b>${n0(ref.alerta)}</b>, inundação <b>${n0(ref.inundacao)}</b> cm. Agora: <b>${status}</b>${statusPrev && statusPrev !== status ? `; previsto: <b>${statusPrev}</b>` : ""}.</div>`
          : `<div class="ref-line">Cotas de referência ainda não confirmadas para esta estação (a régua mudou depois das cheias de 2024).</div>`}
    ${e.fora_da_faixa && e.fora_da_faixa.length ? `<div class="ref-line">Algumas entradas estão fora da faixa vista no treino; a previsão foi limitada a essa faixa.</div>` : ""}
    <svg id="now-chart-${c}" role="img" aria-label="Cota de ${esc(e.nome)} nas últimas 72 horas e prevista"></svg>`;

  const box = el.getBoundingClientRect(), W = Math.max(280, box.width - 28), Hh = 210, m = { t: 14, r: 12, b: 24, l: 48 };
  const svg = d3.select(`#now-chart-${c}`).attr("viewBox", `0 0 ${W} ${Hh}`).attr("width", "100%").attr("height", Hh);
  const obs = e.obs || [], pts = T.map((t, i) => [t, obs[i]]).filter(d => d[1] != null);
  const fpts = tr && e.cota_t != null ? [[tr, e.cota_t], ...fcs.map(q => [utc(q.t), q.cota])] : [];
  const vals = pts.map(d => d[1]).concat(fpts.map(d => d[1]));
  if (!vals.length) return;
  const x1 = fpts.length ? fpts.at(-1)[0] : T.at(-1);
  const x = d3.scaleUtc().domain([T[0], x1]).range([m.l, W - m.r]);
  let lo = d3.min(vals), hi = d3.max(vals); const pad = Math.max(40, (hi - lo) * .15);
  const refs = ref ? REF_NOMES.map(([k, nome]) => ({ v: ref[k], nome })).filter(r => r.v >= lo - pad && r.v <= hi + 3 * pad) : [];
  hi = Math.max(hi + pad, ...refs.map(r => r.v + pad / 2)); lo = Math.max(0, lo - pad);
  const y = d3.scaleLinear().domain([lo, hi]).nice().range([Hh - m.b, m.t]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(""));
  if (tr) svg.append("rect").attr("x", x(tr)).attr("width", Math.max(0, x(x1) - x(tr))).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", css("--silt-soft"));
  refs.forEach(r => {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(r.v)).attr("y2", y(r.v)).attr("stroke", css("--warn")).attr("stroke-dasharray", "6 4").attr("stroke-width", 1.2);
    svg.append("text").attr("x", m.l + 4).attr("y", y(r.v) - 4).style("fill", css("--warn")).text(`${r.nome} ${n0(r.v)}`);
  });
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).ticks(W < 480 ? 4 : 7).tickFormat(fHora));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(n0)).call(g => g.select("path").remove());
  const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));
  const lineD = d3.line().defined(d => d[1] != null).x(d => x(d[0])).y(d => y(d[1]));
  svg.append("path").attr("d", lineD(T.map((t, i) => [t, obs[i]]))).attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 2.2).attr("stroke-linejoin", "round");
  if (fpts.length) {
    svg.append("path").attr("d", line(fpts)).attr("fill", "none").attr("stroke", css("--silt")).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    svg.selectAll(null).data(fpts.slice(1)).join("circle").attr("cx", d => x(d[0])).attr("cy", d => y(d[1])).attr("r", 4).attr("fill", css("--silt")).attr("stroke", css("--surface")).attr("stroke-width", 1.5);
    svg.append("text").attr("x", x(x1) - 4).attr("y", m.t + 10).attr("text-anchor", "end").style("fill", css("--silt-ink")).text("previsão");
  }
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", Hh - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", ev => {
      const [px] = d3.pointer(ev), t = x.invert(px);
      const q = fpts.length && t > tr ? fpts.slice(1).reduce((a, b) => Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a) : null;
      if (q) return showTip(ev, `<div class="t">${fHora(q[0])} (hora local)</div>${row("Previsto", `${n0(q[1])} cm`, css("--silt"))}`);
      const i = Math.max(0, Math.min(T.length - 1, Math.round((t - T[0]) / 36e5)));
      showTip(ev, `<div class="t">${fHora(T[i])} (hora local)</div>${row("Observado", obs[i] == null ? "—" : `${n0(obs[i])} cm`, css("--ink"))}`);
    }).on("mouseleave", hideTip);
}

function spark(sel, obs, T) {
  const svg = d3.select(sel), W = Math.max(200, svg.node().getBoundingClientRect().width || 300), Hh = 34;
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).attr("preserveAspectRatio", "none").selectAll("*").remove();
  const v = obs.filter(d => d != null); if (!v.length) return;
  const x = d3.scaleLinear().domain([0, T.length - 1]).range([2, W - 2]);
  const y = d3.scaleLinear().domain(d3.extent(v)).nice().range([Hh - 3, 3]);
  svg.append("path").attr("d", d3.line().defined(d => d != null).x((d, i) => x(i)).y(d => y(d))(obs))
    .attr("fill", "none").attr("stroke", css("--ink-2")).attr("stroke-width", 1.5).attr("vector-effect", "non-scaling-stroke");
}

function renderNowRain(T) {
  const C = LIVE.chuva, P = LIVE.chuva_prevista, svg = d3.select("#now-rain"), acc = document.getElementById("now-acc");
  const fm = P ? (P.modelos.gfs_seamless || Object.values(P.modelos)[0]) : null;
  if (!C && !fm) { svg.attr("height", 0); acc.innerHTML = `<span class="note">Chuva do IMERG e previsão de chuva indisponíveis nesta atualização.</span>`; return; }
  const box = svg.node().parentNode.getBoundingClientRect(), W = Math.max(280, box.width - 24), Hh = 130, m = { t: 14, r: 10, b: 22, l: 36 };
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).attr("width", "100%").attr("height", Hh).selectAll("*").remove();
  const t0 = C ? utc(C.inicio) : T[0];
  const d = C ? C.bacia.map((v, i) => ({ t: new Date(+t0 + i * 36e5), v })) : [];
  const ult = C && C.ultima_hora ? utc(C.ultima_hora) : null;
  const agora = utc(LIVE.gerado_em), fim = new Date(+agora + 24 * 36e5);
  const f0 = fm ? utc(fm.inicio) : null;
  const fd = fm ? fm.bacia.map((v, i) => ({ t: new Date(+f0 + i * 36e5), v }))
    .filter(q => q.v != null && (!ult || q.t > ult) && q.t >= t0 && q.t < fim) : [];
  const x = d3.scaleUtc().domain([t0, fm ? fim : new Date(+t0 + d.length * 36e5)]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, Math.max(2, d3.max(d.concat(fd), q => q.v) || 0)]).nice().range([Hh - m.b, m.t]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSize(-(W - m.l - m.r)).tickFormat(""));
  if (ult) svg.append("rect").attr("x", x(new Date(+ult + 36e5))).attr("width", Math.max(0, x(agora) - x(new Date(+ult + 36e5)))).attr("y", m.t).attr("height", Hh - m.t - m.b).attr("fill", css("--gap"));
  const bw = Math.max(1, x(new Date(+t0 + 36e5)) - x(t0) - 1);
  const bars = (data, cor, nome, op) => svg.append("g").selectAll("rect").data(data.filter(q => q.v)).join("rect")
    .attr("x", q => x(q.t)).attr("width", bw).attr("y", q => y(q.v)).attr("height", q => y(0) - y(q.v)).attr("fill", cor).attr("opacity", op)
    .on("mousemove", (ev, q) => showTip(ev, `<div class="t">${fHora(q.t)} (hora local)</div>${row(nome, `${n1(q.v)} mm/h`, cor)}`)).on("mouseleave", hideTip);
  bars(d, css("--seq-3"), "IMERG (observado)", 1);
  if (fm) bars(fd, css("--silt"), `${fm.nome} (previsto)`, .75);
  if (fm) {
    svg.append("line").attr("x1", x(agora)).attr("x2", x(agora)).attr("y1", m.t - 4).attr("y2", Hh - m.b).attr("stroke", css("--ink-3")).attr("stroke-dasharray", "3 3");
    svg.append("text").attr("x", x(agora) + 4).attr("y", m.t - 3).style("fill", css("--silt-ink")).text("previsão →");
  }
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).ticks(W < 480 ? 4 : 6).tickFormat(fHora));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickFormat(v => Number.isInteger(v) ? n0(v) : n1(v))).call(g => g.select("path").remove());
  const nomes = { bacia: "bacia inteira", antas: "Antas", taquari_sup: "Taquari sup.", guapore: "Guaporé" };
  let html = C ? Object.entries(C.acumulado_24h).map(([k, v]) => `<div><b>${n1(v)} mm</b><span>${nomes[k] || k} · últimas 24 h (72 h: ${n1(C.acumulado_72h[k])})</span></div>`).join("") : "";
  if (P) html += Object.values(P.modelos).map(v => `<div><b style="color:${css("--silt-ink")}">${n1(v.acumulado_24h.bacia)} mm</b><span>${esc(v.nome)} · próximas 24 h (12 h: ${n1(v.acumulado_12h.bacia)})</span></div>`).join("");
  if (ult) html += `<div><span>Satélite (IMERG) até ${fHora(ult)}; depois disso, as barras são da previsão ${fm ? esc(fm.nome) : ""}.</span></div>`;
  if (P && P.nivel !== "normal") html += `<div><span style="color:${css("--warn")}"><b style="font-size:.9rem;color:inherit">${P.nivel === "alto" ? "Chuva forte prevista" : "Atenção à chuva prevista"}</b> (≥ ${P.limiares_24h[P.nivel === "alto" ? 1 : 0]} mm em 24 h). A cota prevista ainda não considera essa chuva.</span></div>`;
  acc.innerHTML = html;
}

/* ---------------------------------------------------------------- conferência ao vivo */
function renderConferencia() {
  const body = document.getElementById("conf-body");
  const V = LIVE && LIVE.verificacao;
  if (!V || !Object.keys(V.estacoes || {}).length) {
    body.innerHTML = `<div class="panel live-off">${LIVE ? "Ainda não há previsões ao vivo com a hora prevista já medida pela ANA." : "Os dados ao vivo não puderam ser carregados agora."}</div>`;
    return;
  }
  const h = +S.vh, T0 = utc(V.inicio), T = d3.range(V.n).map(i => new Date(+T0 + i * 36e5));
  const codes = Object.keys(V.estacoes).filter(c => LIVE.estacoes[c]);
  body.innerHTML = `<div class="two">${codes.map(c => `<div class="panel conf-card" id="conf-${c}"></div>`).join("")}</div>
    <p class="note" style="margin-top:12px">Métricas acumuladas desde ${fData(utc(V.desde))}, quando a rotina começou a guardar as previsões; o gráfico mostra os últimos ${V.dias} dias. Erro = previsto − observado. A persistência repete a cota do momento da previsão. Enquanto houver poucas semanas de dados, os números mudam muito de um dia para o outro e não substituem a avaliação histórica (seção 05). Horas sem ponto são rodadas que não aconteceram ou horas sem leitura da ANA.</p>`;
  codes.forEach(c => renderConfCard(c, V.estacoes[c], T, h));
}

function renderConfCard(c, e, T, h) {
  const el = document.getElementById(`conf-${c}`), nome = LIVE.estacoes[c].nome;
  const P = (e.pares || []).find(q => q.h === h) || { i: [], prev: [], obs: [], base: [] };
  const M = e.metricas || [], mh = M.find(q => q.h === h);
  const resumo = mh ? `Em +${h} h, o erro típico (RMSE) foi de <b>${n0(mh.rmse)} cm</b>, contra <b>${n0(mh.rmse_persist)} cm</b> da persistência${mh.skill == null ? "" : ` (skill ${n2(mh.skill)})`}, em ${n0(mh.n)} previsões.` : `Ainda sem previsões de +${h} h conferidas.`;
  el.innerHTML = `<h3>${esc(nome)}</h3>
    <div class="ref-line">${resumo}</div>
    <svg id="conf-chart-${c}" role="img" aria-label="Cota observada em ${esc(nome)} nos últimos dias e cota prevista com ${h} horas de antecedência"></svg>
    <div class="tbl-wrap"><table class="conf-tbl"><thead><tr><th>Horizonte</th><th>Previsões</th><th>Erro médio abs.</th><th>Viés</th><th>RMSE</th><th>RMSE persist.</th><th>Skill</th></tr></thead>
      <tbody>${M.map(q => `<tr class="${q.h === h ? "sel" : ""}"><td>+${q.h} h</td><td>${n0(q.n)}</td><td>${n0(q.mae)} cm</td><td>${sgn0(q.vies)} cm</td><td>${n0(q.rmse)} cm</td><td>${n0(q.rmse_persist)} cm</td><td>${q.skill == null ? "—" : n2(q.skill)}</td></tr>`).join("")}</tbody></table></div>`;

  const svg = d3.select(`#conf-chart-${c}`), cs = getComputedStyle(el);
  const W = Math.max(260, el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)), Hh = 220, m = { t: 12, r: 10, b: 24, l: 48 };
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).attr("width", "100%").attr("height", Hh);
  const obs = e.obs || [], pts = P.i.map((i, k) => ({ t: T[i], prev: P.prev[k], obs: P.obs[k], base: P.base[k] })).filter(d => d.t && d.prev != null);
  const vals = obs.filter(v => v != null).concat(pts.map(d => d.prev));
  if (!vals.length) { svg.attr("height", 0); return; }
  const x = d3.scaleUtc().domain([T[0], T.at(-1)]).range([m.l, W - m.r]);
  const lo = d3.min(vals), hi = d3.max(vals), pad = Math.max(30, (hi - lo) * .1);
  const y = d3.scaleLinear().domain([Math.max(0, lo - pad), hi + pad]).nice().range([Hh - m.b, m.t]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Hh - m.b})`).call(d3.axisBottom(x).ticks(d3.utcDay.every(W < 480 ? 2 : 1)).tickFormat(fDia));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(n0)).call(g => g.select("path").remove());
  // fio fino ligando previsto e observado em cada hora: o tamanho do erro fica visível
  svg.append("g").selectAll("line").data(pts.filter(d => d.obs != null)).join("line")
    .attr("x1", d => x(d.t)).attr("x2", d => x(d.t)).attr("y1", d => y(d.obs)).attr("y2", d => y(d.prev))
    .attr("stroke", css("--silt")).attr("stroke-opacity", .45).attr("stroke-width", 1);
  svg.append("path").attr("d", d3.line().defined(d => d != null).x((d, i) => x(T[i])).y(d => y(d))(obs))
    .attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 2).attr("stroke-linejoin", "round");
  svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", d => x(d.t)).attr("cy", d => y(d.prev)).attr("r", 3.2)
    .attr("fill", css("--silt")).attr("stroke", css("--surface")).attr("stroke-width", 1);
  const bis = d3.bisector(d => d.t).center;
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", Hh - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", ev => {
      const t = x.invert(d3.pointer(ev)[0]);
      const i = Math.max(0, Math.min(T.length - 1, Math.round((t - T[0]) / 36e5)));
      const d = pts.length ? pts[bis(pts, t)] : null;
      if (d && Math.abs(d.t - t) <= 1.5 * 36e5) {
        const tr = new Date(+d.t - h * 36e5);
        return showTip(ev, `<div class="t">${fHora(d.t)} (hora local)</div>${row("Observado", d.obs == null ? "—" : `${n0(d.obs)} cm`, css("--ink"))}${row(`Previsto às ${ptTime.utcFormat("%d/%m %Hh")(tr)}`, `${n0(d.prev)} cm`, css("--silt"))}${d.obs == null ? "" : row("Erro", `${sgn0(d.prev - d.obs)} cm`)}${d.obs == null || d.base == null ? "" : row("Erro da persistência", `${sgn0(d.base - d.obs)} cm`)}`);
      }
      showTip(ev, `<div class="t">${fHora(T[i])} (hora local)</div>${row("Observado", obs[i] == null ? "—" : `${n0(obs[i])} cm`, css("--ink"))}`);
    }).on("mouseleave", hideTip);
}

/* ---------------------------------------------------------------- montagem */
function renderAll() { renderAgora(); renderFacts(); renderMapa(); renderStations(); renderChips(); renderEvento(); renderSkill(); renderLong(); }
seg("seg-mapa", [["media_anual", "Média anual"], ...EVENTOS.map(e => [e, e])], "mapa", renderMapa);
seg("seg-est", ALVOS.map(c => [c, EST[c].nome]), "est", renderEvento);
seg("seg-ev", EVENTOS.map(e => [e, e]), "ev", renderEvento);
seg("seg-h", D.horizontes.map(h => [String(h), `${h} h`]), "h", renderEvento);
seg("seg-vh", D.horizontes.map(h => [String(h), `${h} h`]), "vh", renderConferencia);
const lv = document.getElementById("livre"); lv.checked = S.livre; lv.onchange = () => { S.livre = lv.checked; persist(); renderEvento(); };
renderAll();
/* ---------------------------------------------------------------- navegação: seção atual e progresso de leitura */
(function () {
  const links = [...document.querySelectorAll(".site-nav ol a")];
  const alvos = links.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const prog = document.getElementById("nav-progress");
  let raf = 0;
  function atualiza() {
    raf = 0;
    const lim = 90;                                  // altura da barra + folga
    let atual = null;
    alvos.forEach(el => { if (el.getBoundingClientRect().top <= lim) atual = el; });
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) atual = alvos.at(-1);
    links.forEach(a => a.setAttribute("aria-current", String(!!atual && a.getAttribute("href") === `#${atual.id}`)));
    const at = links.find(a => a.getAttribute("aria-current") === "true");
    if (at && at.parentElement.parentElement.scrollWidth > at.parentElement.parentElement.clientWidth)
      at.scrollIntoView({ block: "nearest", inline: "nearest" });
    const max = document.documentElement.scrollHeight - innerHeight;
    prog.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
  }
  addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(atualiza); }, { passive: true });
  addEventListener("resize", atualiza);
  atualiza();
})();
loadLive();
setInterval(loadLive, 10 * 60e3);
let rt; new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(renderAll, 120); }).observe(document.querySelector(".wrap"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
