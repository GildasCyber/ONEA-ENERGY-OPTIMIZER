/* ================================================================
   PASPANGA DASHBOARD — dashboard.js v2.0
   5 onglets : Vue d'ensemble, Prédiction, Optimisation,
               Anomalies, Classement
================================================================ */
'use strict';

// ── Palette Chart.js ──────────────────────────────────────────
const C = {
  blue:       '#3b82f6',
  bluePale:   'rgba(59,130,246,0.12)',
  blueLight:  '#60a5fa',
  green:      '#22c55e',
  greenPale:  'rgba(34,197,94,0.12)',
  orange:     '#f97316',
  orangePale: 'rgba(249,115,22,0.12)',
  teal:       '#22d3ee',
  red:        '#ef4444',
  yellow:     '#eab308',
  yellowBg:   '#eab308',
  gray400:    '#64748b',
  gray200:    'rgba(255,255,255,0.07)',
  gray700:    '#94a3b8',
  white:      '#f1f5f9',
};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 12 } },
    tooltip: {
      backgroundColor: '#0d1626',
      titleColor: '#f1f5f9',
      bodyColor: '#94a3b8',
      borderColor: '#1e2d42',
      borderWidth: 1,
      padding: 10,
      titleFont: { family: 'Inter', size: 12, weight: '600' },
      bodyFont: { family: 'JetBrains Mono', size: 11 },
    },
  },
  scales: {
    x: {
      ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11, weight: '500' }, maxRotation: 0 },
      grid:  { color: C.gray200 },
      border: { color: 'rgba(255,255,255,0.07)' },
    },
    y: {
      ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11, weight: '500' } },
      grid:  { color: C.gray200 },
      border: { color: 'rgba(255,255,255,0.07)' },
    },
  },
};

// ── Charts instances ──────────────────────────────────────────
const charts = {};

// ── State ─────────────────────────────────────────────────────
let scheduleData    = [];
let predictionsData = [];
let kpiState        = {};
let realtimeState = {};
let realtimeHistory = [];

// ── Utilitaires ───────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = (v, d=0) => v == null || isNaN(v) ? '--' : Number(v).toLocaleString('fr-FR', { minimumFractionDigits:d, maximumFractionDigits:d });
const set = (id, v) => { const e=$(id); if(e) e.textContent=v; };
const setHtml = (id, v) => { const e=$(id); if(e) e.innerHTML=v; };

function isGE(src) {
  return src === 'GROUPE' || src === 'GENERATOR';
}

function srcBadge(src) {
  if (src === 'SOLAR')   return '<span class="src-badge solar">☀ Solaire</span>';
  if (src === 'SONABEL') return '<span class="src-badge sonabel">⚡ SONABEL</span>';
  if (isGE(src))         return '<span class="src-badge ge">⛽ GE</span>';
  return `<span class="src-badge">${src||'--'}</span>`;
}

function mkChart(id, type, data, options={}) {
  const ctx = $(id);
  if (!ctx) return null;
  if (charts[id]) { charts[id].destroy(); }
  charts[id] = new Chart(ctx, {
    type,
    data,
    options: deepMerge(CHART_DEFAULTS, options),
  });
  return charts[id];
}

function deepMerge(a, b) {
  const r = Object.assign({}, a);
  for (const k in b) {
    if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]))
      r[k] = deepMerge(a[k]||{}, b[k]);
    else r[k] = b[k];
  }
  return r;
}

// ── Horloge ───────────────────────────────────────────────────
function tickClock() {
  const n = new Date();
  set('clock', n.toLocaleTimeString('fr-FR'));
  set('hdate', n.toLocaleDateString('fr-FR', { weekday:'short', day:'2-digit', month:'short', year:'numeric' }).toUpperCase());
}
setInterval(tickClock, 1000); tickClock();

// ── Navigation onglets ────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const tab = $(`tab-${btn.dataset.tab}`);
    if (tab) tab.classList.add('active');
    // Rendre les graphiques de l'onglet actif
    if (btn.dataset.tab === 'prediction')   renderPredictionTab();
    if (btn.dataset.tab === 'optimisation') renderOptimisationTab();
    if (btn.dataset.tab === 'anomalies')    renderAnomaliesTab();
  });
});

// ═══════════════════════════════════════════════════════════════
// VUE D'ENSEMBLE
// ═══════════════════════════════════════════════════════════════

function renderOverview(kpi) {
  kpiState = kpi;
  // KPI cards
  set('k-cout', fmtKPI(kpi.cout_total_fcfa, 'FCFA', 0));
  set('k-eco',     `Économie : ${fmtKPI(kpi.economie_fcfa, 'FCFA', 0)} (${kpi.economie_pct}%)`);
  set('k-solar',   fmtKPI(kpi.mix_kwh?.solar, 'kWh', 0));
  set('k-solar-pct', `${kpi.part_solaire_pct} % du mix`);
  function fmtKPI(v, unit = '', d = 1) {
    if (v === null || v === undefined || isNaN(v)) return `-- ${unit}`;
    return `${Number(v).toLocaleString('fr-FR', {
      minimumFractionDigits: d,
      maximumFractionDigits: d
    })} ${unit}`;
  }
  set('k-co2', `CO₂ : ${fmtKPI(kpi.co2_kg, 'kg', 1)}`);
  set('k-gasoil', fmtKPI(kpi.gasoil_liters, 'L', 1));
  set('k-regime',  kpi.current_pump_label || '--');
  set('k-source',  `Source : ${kpi.current_source || '--'}`);
  set('k-ch-moy',  fmt(kpi.chateau_moyen_pct, 1) + ' %');
  set('k-bache',   `Bâches : ${fmt(kpi.bache_niveau_pct, 1)} %`);

  // Châteaux
  const niveaux = realtimeState.chateau_levels_pct_sim || kpi.chateau_niveaux || [0,0,0,0];
  niveaux.forEach((lv, i) => {
    const fill = $(`cf-${i}`);
    const pct  = $(`cp-${i}`);
    const vol  = $(`cv-${i}`);
    if (!fill) return;
    fill.style.height = lv + '%';
    fill.className = 'ch-fill' + (lv >= 90 ? ' full' : lv < 25 ? ' low' : lv < 40 ? ' warning' : '');
    if (pct) pct.textContent = fmt(lv, 0) + '%';
    if (vol) vol.textContent = fmt((lv/100)*2000, 0) + ' m³';
    const arrow = $(`ca-${i}`);
    if (arrow && i < 3) arrow.classList.toggle('active', kpi.current_n_pumps > 0);
  });

  // Bâches
  const bPct = realtimeState.bache_level_pct_sim ?? kpi.bache_niveau_pct ?? 0;
  const bf = $('bache-fill-bar');
  if (bf) {
    bf.style.width = bPct + '%';
    bf.className = 'bache-fill' + (bPct < 20 ? ' danger' : bPct < 35 ? ' warning' : '');
  }
  set('bache-pct-txt', fmt(bPct,1) + '%');
  set('b-min', fmt(kpi.bache_niveau_min,1) + '%');
  set('b-max', fmt(kpi.bache_niveau_max,1) + '%');
  set('b-vol', fmt((bPct/100)*6000,0) + ' m³');

  // Pompes
  const nP = kpi.current_n_pumps || 0;
  [1,2,3,4].forEach(i => {
    const cell = $(`pm-${i}`);
    if (!cell) return;
    const active = i === 4 ? nP === 4 : i <= nP;
    cell.classList.toggle('active', active);
  });
  set('ps-label', kpi.current_pump_label || '--');
  set('ps-power', (kpi.current_power_kw || 0) + ' kW');
  set('ps-flow',  (kpi.current_flow_m3h || 0) + ' m³/h');
  setHtml('ps-src', srcBadge(kpi.current_source));

  // Mix donut
  renderMixDonut(kpi.mix_kwh || {});

  set('last-update', new Date().toLocaleTimeString('fr-FR'));
}

function renderMixDonut(mix) {
  const sol = mix.solar   || 0;
  const son = mix.sonabel || 0;
  const ge  = mix.ge      || 0;
  const tot = sol + son + ge || 1;

  set('mix-total-val', fmt(tot, 0));
  set('mx-sol',   fmt(sol,0) + ' kWh'); set('mx-sol-p', fmt(sol/tot*100,1) + '%');
  set('mx-son',   fmt(son,0) + ' kWh'); set('mx-son-p', fmt(son/tot*100,1) + '%');
  set('mx-ge',    fmt(ge,0)  + ' kWh'); set('mx-ge-p',  fmt(ge/tot*100,1)  + '%');

  const ctx = $('chart-mix');
  if (!ctx) return;
  const d = { datasets:[{ data:[sol||.1,son||.1,ge||.1], backgroundColor:[C.yellowBg,C.blue,C.orange], borderWidth:0, hoverOffset:4 }] };
  if (charts['chart-mix']) { charts['chart-mix'].data=d; charts['chart-mix'].update('none'); return; }
  charts['chart-mix'] = new Chart(ctx, {
    type:'doughnut', data:d,
    options:{ responsive:false, cutout:'70%', plugins:{ legend:{display:false}, tooltip:{enabled:false} }, animation:{duration:600} }
  });
}

function renderScheduleChart(sched) {
  if (!sched.length) return;
  const labels  = sched.map((s,i) => i%2===0 ? (s.datetime?.slice(11,16)||s.hour+'h') : '');
  const nPumps  = sched.map(s => s.n_pumps);
  const bgColors = sched.map(s =>
    s.source==='SOLAR' ? C.yellowBg+'cc' :
    isGE(s.source)     ? C.orange+'cc'   : C.blue+'bb'
  );
  const tarifs = sched.map(s => s.tarif_fcfa_kwh);

  mkChart('chart-schedule', 'bar', {
    labels,
    datasets:[
      { label:'Pompes actives', data:nPumps, backgroundColor:bgColors, borderRadius:2, yAxisID:'yP', order:2 },
      { label:'Tarif FCFA/kWh', data:tarifs, type:'line', borderColor:C.orange, borderWidth:1.5,
        borderDash:[4,3], pointRadius:0, yAxisID:'yT', order:1, fill:false },
    ]
  }, {
    plugins:{ tooltip:{ callbacks:{
      title: items => sched[items[0].dataIndex]?.datetime?.slice(11,16) || '',
      label: item => item.dataset.label==='Pompes actives'
        ? ` ${sched[item.dataIndex].pump_label} · ${sched[item.dataIndex].source}`
        : ` Tarif : ${sched[item.dataIndex].tarif_fcfa_kwh} FCFA/kWh`,
    }}},
    scales:{
      x:{ ticks:{color:'#94a3b8', font:{size:11,weight:'500'}, maxRotation:0}, grid:{color:C.gray200} },
      yP:{ position:'left', min:0, max:4.5, ticks:{stepSize:1, callback:v=>['OFF','1P','2P','3P','4P'][v]||v, font:{size:11,weight:'500',family:'JetBrains Mono'}, color:'#94a3b8'}, grid:{color:C.gray200}, title:{display:true,text:'POMPES',font:{size:11,weight:'500'},color:'#94a3b8'} },
      yT:{ position:'right', min:40, max:130, ticks:{color:C.orange, font:{size:11,weight:'500'}}, grid:{display:false}, title:{display:true,text:'FCFA/kWh',font:{size:11,weight:'500'},color:C.orange} },
    }
  });
}

function renderLevelsChart(sched) {
  if (!sched.length) return;
  const labels  = sched.map((s,i) => i%2===0 ? (s.datetime?.slice(11,16)||s.hour+'h') : '');
  const chMeans = sched.map(s => s.chateau_mean_pct);
  const baches  = sched.map(s => s.bache_level_pct);
  const c1      = sched.map(s => (s.chateau_levels_pct||[0,0,0,0])[0]);
  const c4      = sched.map(s => (s.chateau_levels_pct||[0,0,0,0])[3]);

  mkChart('chart-levels', 'line', {
    labels,
    datasets:[
      { label:'Bâches source', data:baches, borderColor:'#22d3ee', backgroundColor:'rgba(34,211,238,0.08)', fill:true, borderWidth:1.5, pointRadius:0, tension:.3, yAxisID:'y' },
      { label:'Châteaux (moy)', data:chMeans, borderColor:C.blue, fill:false, borderWidth:2, pointRadius:0, tension:.3, yAxisID:'y' },
      { label:'C1 (1er)', data:c1, borderColor:C.yellowBg, fill:false, borderWidth:1, borderDash:[3,2], pointRadius:0, tension:.3, yAxisID:'y' },
      { label:'C4 (dernier)', data:c4, borderColor:C.orange, fill:false, borderWidth:1, borderDash:[3,2], pointRadius:0, tension:.3, yAxisID:'y' },
    ]
  }, {
    scales:{
      x:{ ticks:{color:'#94a3b8',font:{size:11,weight:'500'},maxRotation:0}, grid:{color:C.gray200} },
      y:{ min:0, max:100, ticks:{callback:v=>v+'%',font:{size:11,weight:'500'},color:'#94a3b8'}, grid:{color:C.gray200} },
    }
  });
}


// ═══════════════════════════════════════════════════════════════
// PRÉDICTION
// ═══════════════════════════════════════════════════════════════

function renderPredictionTab() {
  if (!predictionsData.length) return;
  const preds  = predictionsData;
  const labels = preds.map((p,i) => i%4===0 ? `${p.hour}h` : '');
  const H      = [... new Set(preds.map(p=>p.hour))];
  const byHour = h => preds.filter(p=>p.hour===h)[0];

  // Énergie prédite
  mkChart('chart-pred-energy', 'line', {
    labels,
    datasets:[{ label:'Énergie prédite (kW)', data:preds.map(p=>p.energy_predicted),
      borderColor:C.blue, backgroundColor:C.bluePale, fill:true, borderWidth:2, pointRadius:0, tension:.4 }]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}},grid:{color:C.gray200}}, y:{ticks:{font:{size:11,weight:'500'}},grid:{color:C.gray200}} } });

  // Flow
  mkChart('chart-pred-flow', 'line', {
    labels,
    datasets:[{ label:'Consommation eau (m³/h)', data:preds.map(p=>p.flow_demand_forecast),
      borderColor:C.teal, fill:false, borderWidth:2, pointRadius:0, tension:.4 }]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{ticks:{font:{size:11,weight:'500'},callback:v=>v+' m³/h'}} } });
  
  // Solaire prédit
  mkChart('chart-pred-solar', 'bar', {
    labels,
    datasets:[{ label:'Solaire disponible (kW)', data:preds.map(p=>p.solar_capacity_predicted),
      backgroundColor:C.yellowBg+'99', borderRadius:2 }]
  });


  // Tarifs
  const tarifs = preds.map(p => p.energy_price_kwh > 0 ? p.energy_price_kwh : (p.hour>=17?118:54));
  mkChart('chart-pred-tarif', 'line', {
    labels,
    datasets:[{ label:'Tarif SONABEL (FCFA/kWh)', data:tarifs,
      borderColor:C.orange, fill:true, backgroundColor:C.orangePale+'55', borderWidth:2, pointRadius:0, tension:0, stepped:true }]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{min:40,max:130,ticks:{font:{size:11,weight:'500'}}} } });

  // Disponibilité réseau
  const gridData = preds.map(p => p.grid_status_predicted);
  mkChart('chart-pred-grid', 'bar', {
    labels,
    datasets:[{ label:'Réseau disponible', data:gridData,
      backgroundColor:preds.map(p=>p.grid_status_predicted?C.green+'99':C.red+'99'), borderRadius:2 }]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{min:0,max:1,ticks:{stepSize:1,callback:v=>v?'OUI':'NON',font:{size:11,weight:'500'}}} } });

  // Tableau
  const tbody = $('tbody-pred');
  if (tbody) {
    tbody.innerHTML = preds.map(p => `
      <tr>
        <td>${p.datetime?.slice(-5)||p.hour+'h'}</td>
        <td>${fmt(p.energy_predicted,1)}</td>
        <td>${fmt(p.flow_demand_forecast,1)}</td>
        <td>${fmt(p.solar_capacity_predicted,1)}</td>
        <td>${p.hour>=17?'118 (pointe)':'54 (creuse)'}</td>
        <td>${p.grid_status_predicted?'✓ Dispo':'✗ Coupure'}</td>
        <td>${srcBadge(p.current_source||'SONABEL')}</td>
      </tr>`).join('');
  }
}


// ═══════════════════════════════════════════════════════════════
// OPTIMISATION
// ═══════════════════════════════════════════════════════════════

function renderOptimisationTab() {
  if (!scheduleData.length) return;
  const sched  = scheduleData;
  const labels = sched.map((s,i) => i%2===0 ? (s.datetime?.slice(11,16)||s.hour+'h') : '');

  // KPI optim
  const metrics = kpiState;
  const N = sched.length || 24;
  set('opt-eco',         fmt(metrics.economie_fcfa) + ' FCFA');
  set('opt-eco-pct',     (metrics.economie_pct||'--') + '% vs référence 2 pompes constantes');
  set('opt-hors-pointe', (N - (metrics.slots_pointe||0)) + ' / ' + N + ' créneaux');
  set('opt-pointe',      `Pointe : ${metrics.slots_pointe||'--'} créneaux`);
  set('opt-ge-slots',    (metrics.slots_ge||metrics.slots_ge===0 ? metrics.slots_ge : '--') + ' / ' + N);
  set('opt-ge-kwh',      fmt(metrics.mix_kwh?.ge,1) + ' kWh GE');
  set('opt-arret',       (metrics.slots_arret_pompes??metrics.slots_arret??'--') + ' / ' + N);

  // Planning pompes + sources (couleurs par source)
  const bgSched = sched.map(s =>
    s.source==='SOLAR'       ? C.yellowBg+'cc' :
    s.is_peak_tariff && s.n_pumps > 0 ? C.orange+'bb' :
    isGE(s.source)           ? C.red+'99' : C.blue+'99'
  );
  mkChart('chart-opt-schedule', 'bar', {
    labels,
    datasets:[
      { label:'Pompes actives', data:sched.map(s=>s.n_pumps), backgroundColor:bgSched, borderRadius:2, yAxisID:'yP', order:2 },
      { label:'Cible niveau (%)', data:sched.map(s=>s.target_level_pct), type:'line',
        borderColor:C.green, borderDash:[4,3], borderWidth:1.5, pointRadius:0, yAxisID:'yL', fill:false, order:1 },
    ]
  }, {
    plugins:{ legend:{display:true}, tooltip:{callbacks:{
      title: items => sched[items[0].dataIndex]?.datetime?.slice(11,16)||'',
      label: item => item.dataset.label==='Pompes actives'
        ? ` ${sched[item.dataIndex].pump_label} · ${sched[item.dataIndex].source} · ${fmt(sched[item.dataIndex].cost_fcfa,0)} FCFA`
        : ` Cible : ${item.raw}%`,
    }}},
    scales:{
      x:{ticks:{color:'#94a3b8',font:{size:11,weight:'500'},maxRotation:0},grid:{color:C.gray200}},
      yP:{position:'left',min:0,max:4.5,ticks:{stepSize:1,callback:v=>['OFF','1P','2P','3P','4P'][v]||v,font:{size:11,weight:'500'},color:'#94a3b8'},grid:{color:C.gray200},title:{display:true,text:'POMPES',font:{size:11,weight:'500'},color:'#94a3b8'}},
      yL:{position:'right',min:0,max:100,ticks:{callback:v=>v+'%',font:{size:11,weight:'500'},color:C.green},grid:{display:false},title:{display:true,text:'NIVEAU %',font:{size:11,weight:'500'},color:C.green}},
    }
  });

  // Distribution régimes (doughnut)
  const dist  = metrics.pump_regime_dist || {};
  const rKeys = Object.keys(dist).filter(k=>dist[k]>0);
  const rCols = ['#2d3f58',C.blueLight,'#1d4ed8',C.blue,'#1e3a5f'];
  const rLabs = ['Arrêt','1×90kW','2×90kW','3×90kW','4 pompes'];
  mkChart('chart-opt-regimes', 'doughnut',
    { datasets:[{ data:rKeys.map(k=>dist[k]), backgroundColor:rKeys.map(k=>rCols[+k]||C.blue), borderWidth:2, borderColor:'#141d2e' }],
      labels:rKeys.map(k=>rLabs[+k]||k) },
    { plugins:{ legend:{position:'right',labels:{color:'#94a3b8',font:{size:11},padding:8}}, tooltip:{callbacks:{ label: ctx => ` ${ctx.label} : ${ctx.raw} créneaux (${fmt(ctx.raw/N*100,1)}%)` }} },
      scales:{x:{display:false},y:{display:false}} }
  );

  // Niveaux châteaux
  mkChart('chart-opt-chateaux', 'line', {
    labels,
    datasets:[
      { label:'C1', data:sched.map(s=>(s.chateau_levels_pct||[])[0]||0), borderColor:'#3b82f6', fill:false, borderWidth:1.5, pointRadius:0, tension:.3 },
      { label:'C2', data:sched.map(s=>(s.chateau_levels_pct||[])[1]||0), borderColor:'#60a5fa', fill:false, borderWidth:1.5, pointRadius:0, tension:.3 },
      { label:'C3', data:sched.map(s=>(s.chateau_levels_pct||[])[2]||0), borderColor:'#22d3ee', fill:false, borderWidth:1.5, pointRadius:0, tension:.3 },
      { label:'C4', data:sched.map(s=>(s.chateau_levels_pct||[])[3]||0), borderColor:'#ca8a04', fill:false, borderWidth:1.5, pointRadius:0, tension:.3 },
      { label:'Moy.', data:sched.map(s=>s.chateau_mean_pct||0), borderColor:C.green, fill:false, borderWidth:2, borderDash:[5,3], pointRadius:0, tension:.3 },
    ]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{min:0,max:100,ticks:{callback:v=>v+'%',font:{size:11,weight:'500'}}} } });

  // Niveaux bâches
  mkChart('chart-opt-baches', 'line', {
    labels,
    datasets:[
      { label:'Niveau bâches (%)', data:sched.map(s=>s.bache_level_pct), borderColor:C.teal,
        backgroundColor:'rgba(34,211,238,0.08)', fill:true, borderWidth:2, pointRadius:0, tension:.4 },
    ]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{min:0,max:100,ticks:{callback:v=>v+'%',font:{size:11,weight:'500'}}} } });

  // Coût par créneau
  const coutColors = sched.map(s =>
    isGE(s.source)       ? C.red+'bb' :
    s.is_peak_tariff     ? C.orange+'99' :
    s.source === 'SOLAR' ? C.yellowBg+'99' : C.blue+'88'
  );
  mkChart('chart-opt-cout', 'bar', {
    labels,
    datasets:[{ label:'Coût (FCFA)', data:sched.map(s=>s.cost_fcfa), backgroundColor:coutColors, borderRadius:2 }]
  }, { scales:{ x:{ticks:{font:{size:11,weight:'500'}}}, y:{ticks:{font:{size:11,weight:'500'},callback:v=>fmt(v,0)}} } });

  // Table planning
  const tbody = $('tbody-opt');
  if (tbody) {
    tbody.innerHTML = sched.map(s => {
      let rowClass = '';
      if (isGE(s.source) || s.generator_kw > 0.1) rowClass = 'row-ge';
      else if (s.source === 'SOLAR')               rowClass = 'row-solar';
      else if (s.is_peak_tariff && s.n_pumps > 0)  rowClass = 'row-peak';
      return `<tr class="${rowClass}">
        <td>${s.slot}</td>
        <td>${s.datetime?.slice(11,16)||s.hour+'h'}</td>
        <td>${s.n_pumps}P</td>
        <td>${s.power_kw} kW</td>
        <td>${s.flow_m3h} m³/h</td>
        <td>${srcBadge(s.source)}</td>
        <td>${fmt(s.solar_kw,1)}</td>
        <td>${fmt(s.sonabel_kw,1)}</td>
        <td>${fmt(s.generator_kw,1)}</td>
        <td>${fmt(s.cost_fcfa,0)} FCFA</td>
        <td>${fmt(s.bache_level_pct,1)}%</td>
        <td>${fmt(s.chateau_mean_pct,1)}%</td>
      </tr>`;
    }).join('');
  }
}


// ═══════════════════════════════════════════════════════════════
// FETCH & INIT
// ═══════════════════════════════════════════════════════════════

async function fetchAll() {
  try {
    const [kpiRes, schedRes, predRes] = await Promise.all([
      fetch('/api/kpi'),
      fetch('/api/schedule'),
      fetch('/api/predictions'),
    ]);

    const kpi   = await kpiRes.json();
    const sched = await schedRes.json();
    const preds = await predRes.json();

    await fetchRealtime();

    if (kpi.error) throw new Error(kpi.error);

    kpiState        = kpi;
    scheduleData    = Array.isArray(sched) ? sched : [];
    predictionsData = Array.isArray(preds) ? preds : [];

    renderOverview(kpi);
    renderScheduleChart(scheduleData);
    renderLevelsChart(scheduleData);

    // Si onglet actif => rafraîchir
    const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab;
    if (activeTab === 'prediction')   renderPredictionTab();
    if (activeTab === 'optimisation') renderOptimisationTab();

    // Anomalies (fetch séparé car source différente)
    fetchAnomaliesData();
    fetchRankingPreview();
    fetchSystemStatus();

  } catch(e) {
    console.error('Fetch error:', e);
  }
}

async function fetchRealtime() {
  try {
    const [stateRes, histRes] = await Promise.all([
      fetch('/api/realtime'),
      fetch('/api/realtime/history'),
    ]);

    realtimeState   = await stateRes.json();
    realtimeHistory = await histRes.json();

  } catch (e) {
    console.error('Realtime fetch error:', e);
  }
}

async function fetchSystemStatus() {
  try {
    const res  = await fetch('/api/system');
    const data = await res.json();
    const dot  = $('mpc-status-dot');
    const txt  = $('mpc-status-txt');
    if (dot) {
      dot.className = 'mpc-status-dot ' + (data.status === 'success' ? 'ok' : data.status === 'error' ? 'err' : 'idle');
    }
    if (txt) {
      txt.textContent = data.status === 'error'
        ? 'MPC erreur'
        : data.last_mpc_run
          ? 'MPC ' + data.last_mpc_run.slice(11,16)
          : 'MPC en attente';
    }
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════════
// ANOMALIES
// ═══════════════════════════════════════════════════════════════

let anomaliesData  = [];
let anFilterActive = 'ALL';

const ALERT_LABELS = {
  GASPILLAGE_GASOIL_GROUPE_ELECTROGENE: 'Gaspillage gasoil GE',
  SOLAIRE_NON_EXPLOITE:                 'Solaire non exploité',
  RENDEMENT_SOLAIRE_ANORMAL:            'Rendement solaire anormal',
  NIVEAU_BACHE_CRITIQUE:                'Bâche critique',
  NIVEAU_BACHE_BAS:                     'Bâche basse',
  NIVEAU_CHATEAU_CRITIQUE:              'Château critique',
  NIVEAU_CHATEAU_BAS:                   'Château bas',
  POMPAGE_HEURE_POINTE_SONABEL:         'Pointe SONABEL',
  DEBIT_ANORMAL_POMPE_ACTIVE:           'Débit anormal',
  SURCONSOMMATION_ENERGETIQUE:          'Surconso. énergie',
  COMPORTEMENT_INHABITUEL_ML:           'ML inhabituel',
  FUITE_PROBABLE:                       '💧 Fuite probable',
};

function alertTag(code) {
  const label = ALERT_LABELS[code] || code;
  const isML  = code === 'COMPORTEMENT_INHABITUEL_ML';
  return `<span class="alert-tag${isML?' ml':''}">${label}</span>`;
}

function methodBadge(methods) {
  if (!methods || !methods.length) return '';
  const hasRule = methods.includes('RULE_BASED');
  const hasML   = methods.includes('MACHINE_LEARNING');
  if (hasRule && hasML) return '<span class="method-badge both">Règles + ML</span>';
  if (hasML)            return '<span class="method-badge ml">ML</span>';
  return                       '<span class="method-badge rule">Règles</span>';
}

function scorePill(score) {
  const cls = score >= 6 ? 'hi' : score >= 3 ? 'mid' : 'lo';
  return `<span class="score-pill ${cls}">${score}</span>`;
}

function renderAnomaliesTab() {
  if (!anomaliesData.length) return;

  const all   = anomaliesData;
  const filt  = anFilterActive === 'ALL' ? all : all.filter(a => a.severity === anFilterActive);

  // ── KPI ──────────────────────────────────────────────────────
  const crit = all.filter(a => a.severity === 'CRITIQUE').length;
  const moy  = all.filter(a => a.severity === 'MOYENNE').length;
  const faib = all.filter(a => a.severity === 'FAIBLE').length;
  set('an-total', all.length);
  set('an-crit',  crit);
  set('an-moy',   moy);
  set('an-faib',  faib);
  set('an-periode', `Horizon ${all.length} créneaux`);
  set('an-filter-count', `${filt.length} anomalie${filt.length>1?'s':''} affichée${filt.length>1?'s':''}`);

  // ── Graphique types ───────────────────────────────────────────
  const typeCounts = {};
  all.forEach(a => (a.alerts||[]).forEach(code => {
    const lbl = ALERT_LABELS[code] || code;
    typeCounts[lbl] = (typeCounts[lbl] || 0) + 1;
  }));
  const typeEntries = Object.entries(typeCounts).sort((a,b) => b[1]-a[1]);
  mkChart('chart-an-types', 'bar', {
    labels: typeEntries.map(e => e[0]),
    datasets: [{ label:'Occurrences', data: typeEntries.map(e => e[1]),
      backgroundColor: C.blue+'99', borderRadius: 3 }]
  }, {
    indexAxis: 'y',
    plugins: { legend:{display:false} },
    scales: {
      x: { ticks:{font:{size:11,weight:'500'}}, grid:{color:C.gray200} },
      y: { ticks:{font:{size:11,weight:'500'}, color:'#cbd5e1'}, grid:{display:false} }
    }
  });

  // ── Graphique scores par créneau ──────────────────────────────
  const scoreLabels = all.map(a => a.datetime?.slice(11,16) || a.hour+'h');
  const scoreVals   = all.map(a => a.severity_score);
  const scoreCols   = all.map(a =>
    a.severity === 'CRITIQUE' ? C.red+'cc' :
    a.severity === 'MOYENNE'  ? C.orange+'cc' : C.blue+'88'
  );
  mkChart('chart-an-scores', 'bar', {
    labels: scoreLabels,
    datasets: [{ label:'Score sévérité', data: scoreVals,
      backgroundColor: scoreCols, borderRadius: 2 }]
  }, {
    plugins: { legend:{display:false} },
    scales: {
      x: { ticks:{font:{size:11,weight:'500'}, maxRotation:45} },
      y: { min:0, ticks:{stepSize:2, font:{size:11,weight:'500'}},
           title:{display:true, text:'Score', font:{size:11,weight:'500'}} }
    }
  });

  // ── Table ─────────────────────────────────────────────────────
  const tbody = $('tbody-anomalies');
  if (tbody) {
    tbody.innerHTML = filt.map(a => {
      const rowCls = a.severity === 'CRITIQUE' ? 'row-crit' :
                     a.severity === 'MOYENNE'  ? 'row-moy'  : '';
      return `<tr class="${rowCls}">
        <td>${a.datetime?.slice(11,16) || a.hour+'h'}</td>
        <td><span class="sev-badge ${a.severity}">${a.severity}</span></td>
        <td>${scorePill(a.severity_score)}</td>
        <td>${(a.alerts||[]).map(alertTag).join('')}</td>
        <td>${methodBadge(a.detection_methods)}</td>
        <td>${fmt(a.flow_estimated,1)}</td>
        <td>${fmt(a.flow_m3h,1)}</td>
        <td>${fmt(a.bache_level_pct,1)}%</td>
        <td>${fmt(a.chateau_mean_pct,1)}%</td>
        <td>${srcBadge(a.source)}</td>
      </tr>`;
    }).join('') || '<tr><td colspan="10" style="text-align:center;color:var(--gray-400);padding:24px">Aucune anomalie pour ce filtre</td></tr>';
  }
}

// ── Filtres ───────────────────────────────────────────────────
document.querySelectorAll('.an-filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.an-filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    anFilterActive = btn.dataset.sev;
    renderAnomaliesTab();
  });
});

async function fetchAnomaliesData() {
  try {
    const res  = await fetch('/api/anomalies');
    const data = await res.json();
    const list = data.anomalies || (Array.isArray(data) ? data : []);
    anomaliesData = list;

    // aperçu dans vue d'ensemble
    setHtml('anomaly-data-preview', list.length
      ? `<p style="margin-top:16px;color:#16a34a">${list.length} anomalie(s) détectée(s)</p>`
      : '');

    // si onglet anomalies actif, rafraîchir
    const activeTab = document.querySelector('.tab-btn.active')?.dataset?.tab;
    if (activeTab === 'anomalies') renderAnomaliesTab();
  } catch(e) {}
}

async function fetchRankingPreview() {
  try {
    const res  = await fetch('/api/ranking');
    const data = await res.json();
    if (data.stations?.length) {
      setHtml('ranking-data-preview', `<p style="margin-top:16px;color:#16a34a">${data.stations.length} station(s) dans la base</p>`);
    }
  } catch(e) {}
}

// ── Recalculer ────────────────────────────────────────────────
$('btn-regen')?.addEventListener('click', async () => {
  const btn = $('btn-regen');
  btn.classList.add('loading');
  btn.textContent = 'Calcul en cours…';
  try {
    const res  = await fetch('/api/regenerate', { method:'POST' });
    const data = await res.json();
    if (data.status === 'ok') await fetchAll();
  } catch(e) { console.error(e); }
  finally {
    btn.classList.remove('loading');
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Recalculer`;
  }
});

// ── Démarrage ─────────────────────────────────────────────────
fetchAll();
setInterval(fetchAll, 5 * 60 * 1000);
setInterval(fetchRealtime, 60 * 1000);