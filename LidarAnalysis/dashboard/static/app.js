/* Foveated 2.5D semantic LiDAR map - dashboard front end.
 *
 * Cells arrive as one column-major Float32Array per frame (see
 * foveated/export.py).  Column offsets come from the manifest so the
 * layout is never hard-coded here.
 */
'use strict';

const CSS = getComputedStyle(document.documentElement);
const C = n => CSS.getPropertyValue(n).trim();

/* ---- colour roles -------------------------------------------------- */
// Three super-classes: this is the validated all-pairs-safe encoding and
// it is also exactly the taxonomy the brief asks for.
const KIND_COLOR = { 0: '#6e6e76', 1: '#00BFFF', 2: '#9932CC', 3: '#ff4d4d' };
const KIND_NAME  = { 0: 'Unknown', 1: 'Drivable terrain', 2: 'Static obstacle', 3: 'Dynamic object' };

// Fine classes cannot clear the all-pairs CVD floor at six hues, so this
// mode always ships a legend *and* direct labels on every detected object.
const CLASS_COLOR = {
  0: '#6e6e76', 1: C('--series-1'), 2: C('--series-7'), 3: C('--series-5'),
  4: C('--series-8'), 6: C('--series-4'), 7: '#4a4f58', 8: C('--series-3')
};
const CLASS_NAME = {
  0: 'Unlabelled', 1: 'Car', 2: 'Truck', 3: 'Bicycle',
  4: 'Pedestrian', 6: 'Guardrail', 7: 'Road', 8: 'Terrain'
};
const LOD_COLOR = [C('--series-1'), C('--series-3'), C('--series-4'), C('--series-2')];

const SEQ = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
const seqColor = t => SEQ[Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))))];
// cost ramp: good -> critical, via the reserved status colours
const COST_STOPS = ['#0ca30c', '#fab219', '#ec835a', '#d03b3b'];

const LAT_STAGES = [
  ['preprocess', C('--series-3'), 'ego filter'],
  ['index',      C('--series-1'), 'ring index'],
  ['aggregate',  C('--series-4'), 'cell reduce'],
  ['layers',     C('--series-7'), '2.5D layers'],
  ['semantics',  C('--series-5'), 'class fuse'],
  ['freespace',  C('--series-2'), 'ray carve'],
  ['terrain',    C('--series-8'), 'terrain'],
  ['instances',  '#6e6e76',       'objects']
];

const OVERLAYS = [
  ['rings',    'LOD rings',     true],
  ['free',     'Free space',    true],
  ['objects',  'Objects',       true],
  ['tracks',   'Velocity',      true],
  ['path',     'Planned path',  true],
  ['curbs',    'Curbs',         false],
  ['clear',    'Low clearance', false]
];

/* ---- state --------------------------------------------------------- */
const S = {
  manifest: null, cols: {}, nCols: 0,
  frame: 0, cells: null, nCells: 0,
  layer: 'kind', view: 'top',
  scale: 9, panX: 0, panY: 0,
  playing: false, timer: null,
  overlays: Object.fromEntries(OVERLAYS.map(([k, , d]) => [k, d])),
  plan: null, hover: null, cache: new Map()
};

const cv = document.getElementById('map');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tooltip');

/* ==================================================================== */
/* boot                                                                  */
/* ==================================================================== */
async function boot() {
  S.manifest = await (await fetch('api/manifest')).json();
  S.manifest.cell_columns.forEach((n, i) => S.cols[n] = i);
  S.nCols = S.manifest.cell_columns.length;

  const m = S.manifest, a = m.aggregate;
  document.getElementById('subtitle').textContent =
    `${m.scenario} · ${m.n_frames} sweeps · ` +
    m.config.levels.map(l => `${(l.res * 100).toFixed(0)}cm ≤${l.r_max}m`).join(' · ');

  buildToggles();
  renderStaticPanels();
  const scrub = document.getElementById('scrub');
  scrub.max = m.n_frames - 1;
  scrub.addEventListener('input', () => goto(+scrub.value));
  document.getElementById('play').addEventListener('click', togglePlay);

  document.querySelectorAll('[data-layer]').forEach(b =>
    b.addEventListener('click', () => {
      S.layer = b.dataset.layer;
      document.querySelectorAll('[data-layer]').forEach(o =>
        o.setAttribute('aria-pressed', String(o === b)));
      renderLegend(); draw();
    }));
  document.querySelectorAll('[data-view]').forEach(b =>
    b.addEventListener('click', () => {
      S.view = b.dataset.view;
      document.querySelectorAll('[data-view]').forEach(o =>
        o.setAttribute('aria-pressed', String(o === b)));
      fitView();
      draw();
    }));

  installMapInteraction();
  window.addEventListener('resize', () => { resize(); draw(); });
  resize();

  // Deep link: #frame=8&layer=elev&view=oblique&overlays=rings,objects
  // Handy for dropping a specific view into a slide or a bug report.
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.has('layer')) S.layer = q.get('layer');
  if (q.has('view')) S.view = q.get('view');
  if (q.has('overlays')) {
    const on = new Set(q.get('overlays').split(','));
    for (const [k] of OVERLAYS) S.overlays[k] = on.has(k);
    document.querySelectorAll('#toggles .chip').forEach((l, i) => {
      const k = OVERLAYS[i][0];
      l.dataset.on = S.overlays[k] ? '1' : '0';
      l.querySelector('input').checked = S.overlays[k];
    });
  }
  document.querySelectorAll('[data-layer]').forEach(o =>
    o.setAttribute('aria-pressed', String(o.dataset.layer === S.layer)));
  document.querySelectorAll('[data-view]').forEach(o =>
    o.setAttribute('aria-pressed', String(o.dataset.view === S.view)));

  renderLegend();
  await goto(Math.min(Math.max(+(q.get('frame') || 0), 0), m.n_frames - 1));
  fitView();
  draw();
  document.getElementById('loading').remove();
}

/* ==================================================================== */
/* frame loading                                                         */
/* ==================================================================== */
async function loadCells(i) {
  if (S.cache.has(i)) return S.cache.get(i);
  const buf = await (await fetch(`api/cells/${i}`)).arrayBuffer();
  const arr = new Float32Array(buf);
  if (S.cache.size > 60) S.cache.clear();
  S.cache.set(i, arr);
  return arr;
}

function fitView() {
  // Frame the sweep rather than trusting a fixed zoom: these scenes vary
  // from a 10 m boxed-in view to 70 m of open road, and the oblique
  // projection has a completely different aspect from the top-down one.
  if (!S.cells) return;
  const get = accessors();
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (let i = 0; i < S.nCells; i++) {
    const [u, v] = projRaw(get.x(i), get.y(i), get.z_max(i));
    if (u < u0) u0 = u; if (u > u1) u1 = u;
    if (v < v0) v0 = v; if (v > v1) v1 = v;
  }
  if (!isFinite(u0)) return;
  const spanU = Math.max(u1 - u0, 10), spanV = Math.max(v1 - v0, 10);
  S.scale = Math.max(1.2, Math.min(90,
    Math.min((S.W - 90) / spanU, (S.H - 150) / spanV)));
  const [ax, ay] = anchor();
  S.panX = S.W / 2 - (ax + (u0 + u1) / 2 * S.scale);
  S.panY = S.H / 2 - (ay + (v0 + v1) / 2 * S.scale);
}

async function goto(i) {
  S.frame = i;
  S.cells = await loadCells(i);
  S.nCells = S.cells.length / S.nCols;
  const fm = S.manifest.frames[i];
  S.plan = fm.plan;
  document.getElementById('scrub').value = i;
  document.getElementById('frameno').textContent =
    `${String(i + 1).padStart(2, '0')} / ${S.manifest.n_frames}`;
  renderFramePanels(fm);
  draw();
  if (i + 1 < S.manifest.n_frames) loadCells(i + 1);   // prefetch
}

function togglePlay() {
  S.playing = !S.playing;
  document.getElementById('play').innerHTML = S.playing ? '&#10073;&#10073;' : '&#9654;';
  if (S.playing) {
    S.timer = setInterval(async () => {
      const n = (S.frame + 1) % S.manifest.n_frames;
      await goto(n);
    }, S.manifest.dt * 1000);
  } else clearInterval(S.timer);
}

/* ==================================================================== */
/* column accessors                                                      */
/* ==================================================================== */
const col = name => { const o = S.cols[name] * S.nCells; return i => S.cells[o + i]; };

function cellColor(i, get) {
  switch (S.layer) {
    case 'kind':  return KIND_COLOR[get.kind(i)] || '#6e6e76';
    case 'class': return CLASS_COLOR[get.class_id(i)] || '#6e6e76';
    case 'lod':   return LOD_COLOR[get.level(i)] || '#6e6e76';
    case 'elev': {
      const z = get.z_max(i);
      return seqColor(Math.max(0, Math.min(1, (z - S.zLo) / Math.max(1e-6, S.zHi - S.zLo))));
    }
    case 'cost': {
      const c = get.cost(i);
      if (c < 0) return '#3a2020';                       // blocked
      const t = Math.max(0, Math.min(1, c / 2.2));
      return COST_STOPS[Math.min(COST_STOPS.length - 1, Math.floor(t * COST_STOPS.length))];
    }
  }
  return '#6e6e76';
}

function accessors() {
  const g = {};
  for (const n of S.manifest.cell_columns) g[n] = col(n);
  return g;
}

/* ==================================================================== */
/* canvas                                                               */
/* ==================================================================== */
function resize() {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  S.W = r.width; S.H = r.height;
}

/* World (x forward, y left, z up) -> screen.
 *
 * Split into a scale-free part and an anchor so that fitView() can frame
 * either projection with the same code: whatever proj() does, projRaw()
 * describes it in metres and the fit only has to solve for scale + pan.
 */
function projRaw(x, y, z) {
  if (S.view === 'top') return [-y, -x];
  // Cabinet oblique: y stays horizontal, forward x is compressed into the
  // screen, z keeps full scale -- so height is read directly off the page
  // and the map is visibly 2.5D rather than a flat raster.
  return [-y, -x * 0.60 - (z + 1.6) * 1.0];
}

function anchor() {
  return S.view === 'top' ? [S.W / 2, S.H / 2] : [S.W / 2, S.H * 0.70];
}

function proj(x, y, z) {
  const [u, v] = projRaw(x, y, z);
  const [ax, ay] = anchor();
  return [ax + u * S.scale + S.panX, ay + v * S.scale + S.panY];
}

function invProj(sx, sy) {   // top-down only
  const [ax, ay] = anchor();
  return [-(sy - ay - S.panY) / S.scale, -(sx - ax - S.panX) / S.scale];
}

function shade(hex, f) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * f), g = Math.round(((n >> 8) & 255) * f),
        b = Math.round((n & 255) * f);
  return `rgb(${r},${g},${b})`;
}

function draw() {
  if (!S.cells) return;
  ctx.clearRect(0, 0, S.W, S.H);
  ctx.fillStyle = C('--surface-1');
  ctx.fillRect(0, 0, S.W, S.H);

  const get = accessors();
  computeElevRange(get);

  if (S.overlays.free) drawFreeSpace();
  if (S.view === 'top') drawCellsTop(get); else drawCellsOblique(get);
  if (S.overlays.rings) drawRings();
  if (S.overlays.curbs) drawFlagged(get, 'curb', C('--status-warning'));
  if (S.overlays.clear) drawFlagged(get, 'low_clearance', C('--status-critical'));
  if (S.overlays.path) drawPath();
  if (S.overlays.objects) drawObjects();
  drawEgo();
  drawScalebar();
}

function computeElevRange(get) {
  if (S.layer !== 'elev') return;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < S.nCells; i++) {
    const z = get.z_max(i);
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  S.zLo = lo; S.zHi = hi;
}

function drawCellsTop(get) {
  const s = S.scale;
  // Coarse first so a fine cell is never hidden by its coarser neighbour
  // at a ring seam.
  for (let lvl = 3; lvl >= 0; lvl--) {
    let last = '';
    for (let i = 0; i < S.nCells; i++) {
      if (get.level(i) !== lvl) continue;
      const r = get.res(i), x = get.x(i), y = get.y(i);
      const [sx, sy] = proj(x + r / 2, y + r / 2, 0);
      const w = r * s;
      if (sx < -w || sy < -w || sx > S.W || sy > S.H) continue;
      const c = cellColor(i, get);
      if (c !== last) { ctx.fillStyle = c; last = c; }
      ctx.fillRect(sx, sy, Math.max(w, 1), Math.max(w, 1));
    }
  }
}

function drawCellsOblique(get) {
  const s = S.scale;
  const order = Array.from({ length: S.nCells }, (_, i) => i)
    .sort((a, b) => get.x(b) - get.x(a));     // far to near
  for (const i of order) {
    const r = get.res(i), x = get.x(i), y = get.y(i);
    const zb = get.z_ground(i);
    const zt = Math.max(get.z_max(i), zb + 0.02);
    const base = cellColor(i, get);
    const h = r / 2;
    const [ax, ay] = proj(x + h, y + h, zt);
    const [bx, by] = proj(x + h, y - h, zt);
    const [cx2, cy2] = proj(x - h, y - h, zt);
    const [dx, dy] = proj(x - h, y + h, zt);
    if (ax < -40 || ax > S.W + 40 || ay > S.H + 60 || cy2 < -60) continue;
    // extruded side toward the viewer
    if (zt - zb > 0.04) {
      const [ex, ey] = proj(x - h, y - h, zb);
      const [fx, fy] = proj(x - h, y + h, zb);
      ctx.fillStyle = shade(base.startsWith('#') ? base : '#6e6e76', 0.55);
      ctx.beginPath();
      ctx.moveTo(dx, dy); ctx.lineTo(cx2, cy2); ctx.lineTo(ex, ey); ctx.lineTo(fx, fy);
      ctx.closePath(); ctx.fill();
    }
    ctx.fillStyle = base;
    ctx.beginPath();
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx2, cy2); ctx.lineTo(dx, dy);
    ctx.closePath(); ctx.fill();
  }
}

function drawFreeSpace() {
  const prof = S.manifest.frames[S.frame].free_profile;
  if (!prof || !prof.length) return;
  const n = prof.length;
  ctx.save();
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const az = (i + 0.5) / n * 2 * Math.PI - Math.PI;
    const r = prof[i];
    const [sx, sy] = proj(r * Math.cos(az), r * Math.sin(az), 0);
    i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(57,135,229,0.10)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(57,135,229,0.38)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

function drawRings() {
  const lv = S.manifest.config.levels;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  lv.forEach((l, i) => {
    const R = l.r_max;
    ctx.strokeStyle = LOD_COLOR[i] + '66';
    ctx.beginPath();
    const pts = [[R, R], [R, -R], [-R, -R], [-R, R]].map(([x, y]) => proj(x, y, 0));
    pts.forEach(([sx, sy], k) => k ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy));
    ctx.closePath(); ctx.stroke();
    if (S.view === 'top') {
      const [lx, ly] = proj(R, 0, 0);
      if (ly > 12 && ly < S.H - 4) {
        ctx.setLineDash([]);
        ctx.fillStyle = LOD_COLOR[i];
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(`${(l.res * 100).toFixed(0)} cm · ${R} m`, S.W / 2 + S.panX, ly - 4);
        ctx.setLineDash([4, 4]);
      }
    }
  });
  ctx.restore();
}

function drawFlagged(get, field, color) {
  const f = get[field], s = S.scale;
  ctx.fillStyle = color;
  for (let i = 0; i < S.nCells; i++) {
    if (!f(i)) continue;
    const r = get.res(i);
    const [sx, sy] = proj(get.x(i) + r / 2, get.y(i) + r / 2, get.z_max(i));
    ctx.fillRect(sx, sy, Math.max(r * s, 2.5), Math.max(r * s, 2.5));
  }
}

function drawPath() {
  const p = S.plan;
  if (!p || !p.path || p.path.length < 2) return;
  ctx.save();
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = p.found ? C('--status-good') : C('--status-warning');
  if (!p.found) ctx.setLineDash([6, 5]);
  ctx.beginPath();
  p.path.forEach(([x, y], i) => {
    const [sx, sy] = proj(x, y, 0.02);
    i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  const [gx, gy] = proj(p.goal[0], p.goal[1], 0.02);
  ctx.fillStyle = p.found ? C('--status-good') : C('--status-warning');
  ctx.beginPath(); ctx.arc(gx, gy, 5, 0, 7); ctx.fill();
  ctx.strokeStyle = C('--surface-1'); ctx.lineWidth = 2; ctx.stroke();
  ctx.restore();
}

function drawObjects() {
  const fm = S.manifest.frames[S.frame];
  // Direct labels are the CVD relief for fine-class hue, and the pairs
  // that fail the all-pairs floor are all dynamic classes -- so every
  // dynamic object is labelled. Extended static structures get one label
  // for their largest segment instead of one per fragment, which
  // otherwise buries the map under a dozen "Guardrail" tags.
  const biggestStatic = new Map();
  for (const o of fm.instances) {
    if (o.kind !== 2) continue;
    const cur = biggestStatic.get(o.class_id);
    if (!cur || o.n_cells > cur.n_cells) biggestStatic.set(o.class_id, o);
  }
  ctx.save();
  ctx.font = 'bold 32px system-ui';
  ctx.textAlign = 'center';
  for (const o of fm.instances) {
    if (o.class_id === 7 || o.class_id === 8) continue;
    const label = o.kind === 3 || biggestStatic.get(o.class_id) === o;
    const c = CLASS_COLOR[o.class_id] || '#fff';
    const hx = o.extent_x / 2, hy = o.extent_y / 2;
    const pts = [[o.cx - hx, o.cy - hy], [o.cx + hx, o.cy - hy],
                 [o.cx + hx, o.cy + hy], [o.cx - hx, o.cy + hy]]
      .map(([x, y]) => proj(x, y, o.z_top));
    ctx.strokeStyle = c; ctx.lineWidth = 3.0;
    ctx.beginPath();
    pts.forEach(([sx, sy], i) => i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy));
    ctx.closePath(); ctx.stroke();

    if (!label) continue;
    const [lx, ly] = proj(o.cx + hx, o.cy, o.z_top);
    const txt = `${o.class_name} ${o.range_m.toFixed(0)}m`;
    ctx.fillStyle = 'rgba(13,15,18,0.85)';
    const w = ctx.measureText(txt).width + 24;
    ctx.fillRect(lx - w / 2, ly - 56, w, 48);
    ctx.fillStyle = c;
    ctx.fillText(txt, lx, ly - 20);
  }
  if (S.overlays.tracks) {
    for (const t of fm.tracks) {
      if (!t.velocity_valid || t.rel_speed < 0.4) continue;
      const [x0, y0] = proj(t.cx, t.cy, 0.05);
      const [x1, y1] = proj(t.cx + t.vx * 0.9, t.cy + t.vy * 0.9, 0.05);
      ctx.strokeStyle = C('--status-warning'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 7 * Math.cos(a - 0.4), y1 - 7 * Math.sin(a - 0.4));
      ctx.lineTo(x1 - 7 * Math.cos(a + 0.4), y1 - 7 * Math.sin(a + 0.4));
      ctx.closePath();
      ctx.fillStyle = C('--status-warning'); ctx.fill();
    }
  }
  ctx.restore();
}

function drawEgo() {
  const pts = [[2.1, 0], [-1.2, 0.9], [-0.6, 0], [-1.2, -0.9]].map(([x, y]) => proj(x, y, 0.1));
  ctx.beginPath();
  pts.forEach(([sx, sy], i) => i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy));
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.strokeStyle = C('--surface-1'); ctx.lineWidth = 1.5; ctx.stroke();
}

function drawScalebar() {
  const targets = [1, 2, 5, 10, 20, 50, 100];
  const want = 110 / S.scale;
  const m = targets.find(t => t >= want) || 100;
  const el = document.getElementById('scalebar');
  el.querySelector('.bar').style.width = (m * S.scale) + 'px';
  el.querySelector('span').textContent = `${m} m`;
}

/* ==================================================================== */
/* interaction                                                           */
/* ==================================================================== */
function installMapInteraction() {
  let drag = null, moved = 0;

  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.0016);
    S.scale = Math.max(1.2, Math.min(90, S.scale * k));
    draw();
  }, { passive: false });

  cv.addEventListener('pointerdown', e => {
    drag = { x: e.clientX, y: e.clientY, px: S.panX, py: S.panY };
    moved = 0;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', e => {
    if (drag) {
      S.panX = drag.px + (e.clientX - drag.x);
      S.panY = drag.py + (e.clientY - drag.y);
      moved += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
      draw();
      return;
    }
    hoverAt(e);
  });
  cv.addEventListener('pointerup', async e => {
    const wasDrag = moved > 5;
    drag = null;
    cv.releasePointerCapture(e.pointerId);
    if (wasDrag || S.view !== 'top') return;
    const r = cv.getBoundingClientRect();
    const [wx, wy] = invProj(e.clientX - r.left, e.clientY - r.top);
    await replan(wx, wy);
  });
  cv.addEventListener('pointerleave', () => {
    document.getElementById('readout').hidden = true;
  });
  cv.addEventListener('dblclick', e => { e.preventDefault(); fitView(); draw(); });
}

function hoverAt(e) {
  if (!S.cells || S.view !== 'top') return;
  const r = cv.getBoundingClientRect();
  const [wx, wy] = invProj(e.clientX - r.left, e.clientY - r.top);
  const get = accessors();
  let best = -1, bestD = Infinity;
  for (let i = 0; i < S.nCells; i++) {
    const res = get.res(i);
    const dx = Math.abs(get.x(i) - wx), dy = Math.abs(get.y(i) - wy);
    if (dx < res / 2 && dy < res / 2) { best = i; bestD = 0; break; }
    const d = dx + dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  const el = document.getElementById('readout');
  if (best < 0 || bestD > 1.5) { el.hidden = true; return; }
  const clr = get.clearance(best);
  const cost = get.cost(best);
  const rows = [
    ['position', `${get.x(best).toFixed(2)}, ${get.y(best).toFixed(2)} m`],
    ['range', `${Math.hypot(get.x(best), get.y(best)).toFixed(1)} m`],
    ['cell size', `${(get.res(best) * 100).toFixed(0)} cm (L${get.level(best)})`],
    ['class', `${CLASS_NAME[get.class_id(best)]} ${(get.conf(best) * 100).toFixed(0)}%`],
    ['ground z', `${get.z_ground(best).toFixed(2)} m`],
    ['obstacle h', `${get.obstacle_h(best).toFixed(2)} m`],
    ['clearance', clr < 0 ? 'open sky' : `${clr.toFixed(2)} m`],
    ['step', `${(get.step(best) * 100).toFixed(1)} cm`],
    ['points', String(get.n_pts(best) | 0)],
    ['traversal', cost < 0 ? 'blocked' : cost.toFixed(2)]
  ];
  el.innerHTML = rows.map(([k, v]) =>
    `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
  el.hidden = false;
}

async function replan(x, y) {
  const el = document.getElementById('terrain-kv');
  el.style.opacity = '0.55';
  try {
    const res = await fetch('api/replan', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: S.frame, goal: [x, y] })
    });
    S.plan = await res.json();
    renderFramePanels(S.manifest.frames[S.frame]);
    draw();
  } finally { el.style.opacity = '1'; }
}

/* ==================================================================== */
/* panels                                                               */
/* ==================================================================== */
function buildToggles() {
  const host = document.getElementById('toggles');
  host.innerHTML = '';
  for (const [key, label, def] of OVERLAYS) {
    const l = document.createElement('label');
    l.className = 'chip';
    l.dataset.on = def ? '1' : '0';
    l.innerHTML = `<input type="checkbox" ${def ? 'checked' : ''}><span class="dot"></span>${label}`;
    l.querySelector('input').addEventListener('change', ev => {
      S.overlays[key] = ev.target.checked;
      l.dataset.on = ev.target.checked ? '1' : '0';
      draw();
    });
    host.appendChild(l);
  }
}

function renderLegend() {
  const el = document.getElementById('legend');
  const item = (c, t) => `<li><span class="sw" style="background:${c}"></span>${t}</li>`;
  if (S.layer === 'kind') {
    el.innerHTML = `<h3>Semantic class</h3><ul>${[1, 2, 3, 0]
      .map(k => item(KIND_COLOR[k], KIND_NAME[k])).join('')}</ul>`;
  } else if (S.layer === 'class') {
    el.innerHTML = `<h3>Fine class</h3><ul>${[7, 8, 1, 2, 3, 4, 6, 0]
      .map(k => item(CLASS_COLOR[k], CLASS_NAME[k])).join('')}</ul>`;
  } else if (S.layer === 'lod') {
    el.innerHTML = `<h3>Cell resolution</h3><ul>${S.manifest.config.levels
      .map((l, i) => item(LOD_COLOR[i], `${(l.res * 100).toFixed(0)} cm · ≤${l.r_max} m`))
      .join('')}</ul>`;
  } else if (S.layer === 'elev') {
    el.innerHTML = `<h3>Surface elevation</h3>
      <div class="ramp" style="background:linear-gradient(90deg,${SEQ.join(',')})"></div>
      <div class="ends"><span id="el-lo">low</span><span id="el-hi">high</span></div>`;
  } else {
    el.innerHTML = `<h3>Traversal cost</h3>
      <div class="ramp" style="background:linear-gradient(90deg,${COST_STOPS.join(',')})"></div>
      <div class="ends"><span>free</span><span>costly</span></div>
      <ul style="grid-template-columns:1fr;margin-top:6px">${item('#3a2020', 'not traversable')}</ul>`;
  }
}

function fmtBytes(b) {
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(1) + ' GiB';
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(1) + ' MiB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KiB';
  return b + ' B';
}

function renderStaticPanels() {
  const a = S.manifest.aggregate;

  /* latency */
  const lat = a.latency_ms;
  document.getElementById('t-fps').textContent = Math.round(a.map_fps);
  document.getElementById('t-fps-sub').textContent =
    `${lat.map_total.median.toFixed(1)} ms median · ${lat.map_total.p95.toFixed(1)} ms p95`;
  const total = LAT_STAGES.reduce((s, [k]) => s + (lat[k]?.median || 0), 0);
  document.getElementById('lat-stack').innerHTML = LAT_STAGES.map(([k, c]) =>
    `<span style="background:${c};width:${((lat[k]?.median || 0) / total * 100).toFixed(2)}%"
      title="${k}"></span>`).join('');
  document.getElementById('lat-legend').innerHTML = LAT_STAGES.map(([k, c, lab]) =>
    `<span><i style="background:${c}"></i>${lab} ${(lat[k]?.median || 0).toFixed(2)}ms</span>`).join('');
  document.getElementById('lat-note').textContent =
    `Sensor runs at ${(1 / S.manifest.dt).toFixed(0)} Hz, so the budget is ` +
    `${(S.manifest.dt * 1000).toFixed(0)} ms/sweep — ` +
    `${(S.manifest.dt * 1000 / lat.map_total.median).toFixed(0)}× headroom. ` +
    `A* replanning (${lat.plan.median.toFixed(0)} ms) is excluded: it runs off the mapping thread.`;

  /* memory */
  const cap = a.memory_design_capacity;
  document.getElementById('t-mem1').textContent = cap.uniform_fine.ratio_vs_foveated.toFixed(0) + '×';
  document.getElementById('t-mem2').textContent = cap.voxel_fine.ratio_vs_foveated.toFixed(0) + '×';

  const cov = a.memory_observed_area || {};
  // These span three orders of magnitude, so a linear bar chart would
  // render the winning row as an invisible sliver and a log bar chart
  // would misstate the ratios. The numbers are the message: table them.
  const rows = [
    ['Foveated 2.5D (ours)', cap.foveated.cells, cap.foveated.bytes, '1×'],
    ['Uniform 2.5D @ 5 cm', cap.uniform_fine.cells, cap.uniform_fine.bytes,
      cap.uniform_fine.ratio_vs_foveated.toFixed(0) + '×'],
    ['Uniform 2.5D @ 40 cm', cap.uniform_coarse.cells, cap.uniform_coarse.bytes,
      cap.uniform_coarse.ratio_vs_foveated.toFixed(2) + '×'],
    ['Uniform 3D voxels @ 5 cm', cap.voxel_fine.cells, cap.voxel_fine.bytes,
      cap.voxel_fine.ratio_vs_foveated.toFixed(0) + '×']
  ];
  document.getElementById('mem-bars').innerHTML = `
    <table class="kv memtable">
      <tr><th>representation</th><th>cells</th><th>memory</th><th>vs ours</th></tr>
      ${rows.map(([lab, cells, b, ratio], i) => `
        <tr${i === 0 ? ' class="hi"' : ''}>
          <td>${lab}</td>
          <td>${cells >= 1e9 ? (cells / 1e9).toFixed(2) + ' B' :
               cells >= 1e6 ? (cells / 1e6).toFixed(1) + ' M' :
               cells.toLocaleString()}</td>
          <td>${fmtBytes(b)}</td><td>${ratio}</td>
        </tr>`).join('')}
    </table>`;
  document.getElementById('mem-note').innerHTML =
    `Fully-populated ${cap.extent_m} × ${cap.extent_m} m map — the capacity of the data structure itself. ` +
    (cov.uniform_fine ? `Measured over just the area this sensor actually observed, the saving against a
      uniform 5 cm grid is <strong>${cov.uniform_fine.ratio_vs_foveated}×</strong>
      (${fmtBytes(cov.foveated.bytes)} vs ${fmtBytes(cov.uniform_fine.bytes)} per sweep).` : '');

  /* fidelity charts */
  const f = a.fidelity;
  lineChart('chart-acc', 'legend-acc', [
    { name: 'Foveated', color: C('--series-1'), pts: f.foveated.by_range.map(r => [mid(r), r.class_acc * 100]) },
    { name: 'Uniform 5 cm', color: C('--series-3'), pts: f.uniform_fine?.by_range.map(r => [mid(r), r.class_acc * 100]) },
    { name: 'Uniform 40 cm', color: C('--series-2'), pts: f.uniform_coarse?.by_range.map(r => [mid(r), r.class_acc * 100]) }
  ], { yLabel: 'round-trip class accuracy (%)', unit: '%', decimals: 1, clampHi: 100 });

  lineChart('chart-rmse', 'legend-rmse', [
    { name: 'Foveated', color: C('--series-1'), pts: f.foveated.by_range.map(r => [mid(r), r.xy_rmse * 100]) },
    { name: 'Uniform 5 cm', color: C('--series-3'), pts: f.uniform_fine?.by_range.map(r => [mid(r), r.xy_rmse * 100]) },
    { name: 'Uniform 40 cm', color: C('--series-2'), pts: f.uniform_coarse?.by_range.map(r => [mid(r), r.xy_rmse * 100]) }
  ], { yLabel: 'horizontal quantisation RMSE (cm)', unit: ' cm', decimals: 1, clampLo: 0 });
}

const mid = r => (r.r_lo + r.r_hi) / 2;

function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}

function lineChart(svgId, legId, series, opt) {
  const svg = document.getElementById(svgId);
  series = series.filter(s => s.pts && s.pts.length);
  if (!series.length) return;
  const W = 340, H = 150, L = 40, R = 8, T = 16, B = 30;
  const xs = series.flatMap(s => s.pts.map(p => p[0]));
  const ys = series.flatMap(s => s.pts.map(p => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.15 || 1;
  y0 -= pad; y1 += pad;
  // Clamp to what the quantity can physically be: an accuracy axis that
  // runs to 100.34 % or an RMSE axis that dips to -0.3 cm is a chart
  // asserting something impossible.
  if (opt.clampLo !== undefined) y0 = Math.max(y0, opt.clampLo);
  if (opt.clampHi !== undefined) y1 = Math.min(y1, opt.clampHi);
  // snap to a round step so the ticks read cleanly
  const step = niceStep((y1 - y0) / 4);
  y0 = Math.floor(y0 / step) * step;
  y1 = Math.ceil(y1 / step) * step;
  if (opt.clampLo !== undefined) y0 = Math.max(y0, opt.clampLo);
  if (opt.clampHi !== undefined) y1 = Math.min(y1, opt.clampHi);
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R);
  const Y = v => H - B - (v - y0) / (y1 - y0 || 1) * (H - T - B);

  const ticks = Math.max(2, Math.min(6, Math.round((y1 - y0) / step)));
  let g = '';
  for (let i = 0; i <= ticks; i++) {
    const v = y0 + (y1 - y0) * i / ticks, y = Y(v);
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="${C('--grid-line')}" stroke-width="1"/>`;
    g += `<text x="${L - 6}" y="${y + 3.5}" text-anchor="end" font-size="9.5" fill="${C('--text-muted')}">${v.toFixed(opt.decimals)}</text>`;
  }
  const xt = series[0].pts.map(p => p[0]);
  for (const v of xt) {
    g += `<text x="${X(v)}" y="${H - B + 13}" text-anchor="middle" font-size="9.5" fill="${C('--text-muted')}">${v.toFixed(0)}</text>`;
  }
  g += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle" font-size="9.5" fill="${C('--text-muted')}">range from sensor (m)</text>`;
  g += `<text x="${L}" y="${T - 5}" font-size="9.5" fill="${C('--text-muted')}">${opt.yLabel}</text>`;

  for (const s of series) {
    const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    g += s.pts.map(p =>
      `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" fill="${s.color}"
        stroke="${C('--surface-2')}" stroke-width="1.5"/>`).join('');
  }
  // hover targets, one column per range band
  g += xt.map(v => {
    const rows = series.map(s => {
      const p = s.pts.find(q => q[0] === v);
      return p ? `<div class="r">${s.name}: ${p[1].toFixed(opt.decimals)}${opt.unit}</div>` : '';
    }).join('');
    return `<rect class="hit" x="${X(v) - 12}" y="${T}" width="24" height="${H - T - B}"
      fill="transparent" data-t="${v.toFixed(0)} m from sensor" data-rows="${encodeURIComponent(rows)}"/>`;
  }).join('');
  svg.innerHTML = g;

  svg.querySelectorAll('.hit').forEach(el => {
    el.addEventListener('pointerenter', ev => {
      tip.innerHTML = `<div class="t">${el.dataset.t}</div>${decodeURIComponent(el.dataset.rows)}`;
      tip.style.opacity = '1';
      tip.style.left = Math.min(ev.clientX + 12, innerWidth - 250) + 'px';
      tip.style.top = (ev.clientY + 12) + 'px';
    });
    el.addEventListener('pointerleave', () => tip.style.opacity = '0');
  });

  document.getElementById(legId).innerHTML = series.map(s =>
    `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
}

function renderFramePanels(fm) {
  document.getElementById('t-cells').textContent = fm.n_cells.toLocaleString();
  document.getElementById('t-cells-sub').textContent =
    `${fm.points.n_kept.toLocaleString()} points · ${fmtBytes(fm.memory.foveated_2_5d.bytes)}`;

  const lv = S.manifest.config.levels;
  const maxc = Math.max(...fm.cells_per_level, 1);
  document.getElementById('ring-bars').innerHTML = fm.cells_per_level.map((n, i) => `
    <div class="barrow">
      <div class="top"><span class="lab">${(lv[i].res * 100).toFixed(0)} cm · ${lv[i].r_min}–${lv[i].r_max} m</span>
        <span class="num">${n.toLocaleString()}</span></div>
      <div class="track"><div class="fill" style="width:${(n / maxc * 100).toFixed(1)}%;background:${LOD_COLOR[i]}"></div></div>
    </div>`).join('');

  const objs = fm.instances.filter(o => o.class_id !== 7 && o.class_id !== 8);
  document.getElementById('obj-count').textContent = `(${objs.length})`;
  const trackOf = c => fm.tracks.find(t => Math.abs(t.cx - c.cx) < 0.01 && Math.abs(t.cy - c.cy) < 0.01);
  document.getElementById('objlist').innerHTML = objs
    .sort((a, b) => a.range_m - b.range_m)
    .map(o => {
      const t = trackOf(o);
      const sp = t && t.velocity_valid && t.rel_speed > 0.4
        ? ` · ${t.rel_speed.toFixed(1)} m/s rel` : '';
      return `<div class="obj">
        <span class="sw" style="background:${CLASS_COLOR[o.class_id]}"></span>
        <span class="nm">${o.class_name}</span>
        <span class="mt">${o.range_m.toFixed(1)} m · ${o.extent_x.toFixed(1)}×${o.extent_y.toFixed(1)}×${o.height.toFixed(1)} m${sp}</span>
      </div>`;
    }).join('') || '<div class="obj"><span class="nm">none in this sweep</span></div>';

  const p = S.plan || fm.plan;
  const c = fm.counts;
  const rows = [
    ['Drivable cells', c.drivable.toLocaleString()],
    ['Static obstacle cells', c.static_cells.toLocaleString()],
    ['Dynamic object cells', c.dynamic_cells.toLocaleString()],
    ['Curb candidates', c.curb.toLocaleString()],
    ['Low-clearance cells', c.low_clearance.toLocaleString()],
    ['Ego returns filtered', fm.points.n_ego_removed.toLocaleString() +
      ` (${(fm.points.ego_fraction * 100).toFixed(0)}%)`],
    ['Path', p.found ? `${p.path.length} waypoints, ${p.level_transitions} ring crossings`
      : (p.reason || 'no path')],
    ['A* expansions', `${p.expanded.toLocaleString()} in ${p.ms.toFixed(0)} ms`]
  ];
  document.getElementById('terrain-kv').innerHTML =
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

boot().catch(e => {
  document.getElementById('loading').textContent = 'failed to load: ' + e.message;
  console.error(e);
});
