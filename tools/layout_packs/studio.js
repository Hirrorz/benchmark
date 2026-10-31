/* ══════════════════════════════════════════════════════════════════════════
 * 变体 A「工作台 Studio」—— 三栏：左区块导航 | 中配置 | 右实时预览
 *
 * 使用方式的变化：从「一路往下滚」变成「左侧导航直达 + 滚动联动高亮」，
 * 右下角常驻校验状态（点一下跳到校验卡）。
 * 只新增一个 <nav>（插在 #appLayout 首位），原有卡片一个都不动。
 * ══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';
  var LP = window.__LP__;
  if (!LP || LP.arch !== 'studio') return;

  var $ = function (id) { return document.getElementById(id); };
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function icon(paths) {
    return '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' + paths + '</svg>';
  }

  var I = {
    common: icon('<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h13M21 17h0"/>' +
                 '<circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="19" cy="17" r="2"/>'),
    resident: icon('<rect x="3" y="4" width="18" height="7" rx="2"/>' +
                   '<rect x="3" y="13" width="18" height="7" rx="2"/>' +
                   '<path d="M7 7.5h.01M7 16.5h.01"/>'),
    queue: icon('<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/>' +
                '<circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>'),
    launch: icon('<circle cx="12" cy="12" r="9"/><path d="M10 8.5l6 3.5-6 3.5z"/>'),
    hit: icon('<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 15l4-5"/><circle cx="12" cy="18" r="1.4"/>'),
    real: icon('<ellipse cx="12" cy="6" rx="8" ry="3"/>' +
               '<path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/>' +
               '<path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'),
    points: icon('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 10v10"/>'),
    perf: icon('<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>'),
    serveEnv: icon('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/>'),
    clientAdv: icon('<rect x="3" y="7" width="18" height="10" rx="5"/><circle cx="9" cy="12" r="3"/>'),
    def: icon('<circle cx="12" cy="12" r="4"/>')
  };

  var SCEN = [
    { k: 'points', t: '多点', ico: I.points },
    { k: 'resident', t: '常驻', ico: I.resident },
    { k: 'launch', t: '直跑', ico: I.launch },
    { k: 'hit', t: '合成', ico: I.hit },
    { k: 'real', t: '真权重', ico: I.real }
  ];

  var rail = null, scenGroup = null, secGroup = null, statusBtn = null;
  var spyBound = false, spyScheduled = false;

  function ensureRail() {
    if (rail && document.body.contains(rail)) return;
    rail = document.createElement('nav');
    rail.id = 'lpStudioRail';
    rail.className = 'lp-rail';
    rail.setAttribute('aria-label', '工作台导航');
    rail.innerHTML =
      '<div class="lp-rail-brand" title="DS4.1 mini benchmark">' +
        '<span class="lp-rail-logo">DS</span><span class="lp-rail-ver">4.1</span>' +
      '</div>' +
      '<div class="lp-rail-cap">场景</div>' +
      '<div class="lp-rail-group" id="lpRailScen" role="group" aria-label="场景"></div>' +
      '<div class="lp-rail-sep" role="presentation"></div>' +
      '<div class="lp-rail-cap">分区</div>' +
      '<div class="lp-rail-group" id="lpRailSecs" role="group" aria-label="配置分区"></div>' +
      '<button type="button" class="lp-rail-status" id="lpRailStatus" title="查看校验结果"></button>';
    LP.layout.insertBefore(rail, LP.layout.firstChild);
    scenGroup = rail.querySelector('#lpRailScen');
    secGroup = rail.querySelector('#lpRailSecs');
    statusBtn = rail.querySelector('#lpRailStatus');
    statusBtn.addEventListener('click', function () {
      var t = $('msgs');
      if (t && t.scrollIntoView) t.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
    buildScen();
  }

  function buildScen() {
    scenGroup.innerHTML = '';
    SCEN.forEach(function (s) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'lp-rail-btn lp-rail-scen';
      b.setAttribute('data-lp-scenario', s.k);
      b.title = s.t;
      b.innerHTML = '<span class="lp-rail-ico">' + s.ico + '</span>' +
                    '<span class="lp-rail-label">' + esc(s.t) + '</span>';
      b.addEventListener('click', function () {
        try { window.setEntry(s.k); } catch (e) { /* noop */ }
        var card = $((LP.primary || {})[s.k]);
        if (card) scrollToCard(card);
      });
      scenGroup.appendChild(b);
    });
  }

  /* 78px 导轨放不下完整标题 —— 用短名（title 里保留全名） */
  var SHORT = {
    common: '通用', residentCard: '常驻', queueCard: '队列', launchCard: '直跑',
    hitCard: '合成', realCard: '真权重', pointsCard: '多点', perfCard: '性能',
    serveEnvCard: '环境', clientAdvCard: '客户端'
  };

  function buildSecs() {
    secGroup.innerHTML = '';
    var list = LP.visibleCards().slice();
    /* points 场景下 CSS 把 #pointsCard 排到最前（order:-1），导航顺序跟随视觉 */
    if (LP.body.dataset.scenario === 'points') {
      var pc = $('pointsCard');
      var i = list.indexOf(pc);
      if (i > 0) { list.splice(i, 1); list.unshift(pc); }
    }
    list.forEach(function (card) {
      var key = LP.cardKey(card);
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'lp-rail-btn lp-rail-sec';
      b.setAttribute('data-lp-key', key);
      b.title = LP.cardTitle(card);
      b.innerHTML = '<span class="lp-rail-ico">' + (I[key] || I.def) + '</span>' +
                    '<span class="lp-rail-label">' +
                    esc(SHORT[key] || LP.cardTitle(card).slice(0, 5)) + '</span>';
      b._card = card;
      b.addEventListener('click', function () { scrollToCard(card); });
      secGroup.appendChild(b);
    });
    spy();
    bindSpy();
  }

  function scrollToCard(card) {
    if (!card) return;
    if (card.scrollIntoView) {
      try { card.scrollIntoView({ block: 'start', behavior: 'smooth' }); }
      catch (e) { card.scrollIntoView(); }
    }
    setActive(LP.cardKey(card));
  }

  function setActive(key) {
    if (!secGroup) return;
    Array.prototype.forEach.call(secGroup.children, function (b) {
      var on = b.getAttribute('data-lp-key') === key;
      b.classList.toggle('on', on);
      if (on) b.setAttribute('aria-current', 'true');
      else b.removeAttribute('aria-current');
    });
  }

  function spy() {
    if (!secGroup) return;
    var items = Array.prototype.slice.call(secGroup.children);
    if (!items.length) return;
    var best = items[0], bestTop = -1e9, measurable = false;
    items.forEach(function (it) {
      var card = it._card;
      if (!card || !card.getBoundingClientRect) return;
      var r = card.getBoundingClientRect();
      if (!r || (r.width === 0 && r.height === 0)) return;   // jsdom：全 0，跳过
      measurable = true;
      if (r.top <= 140 && r.top > bestTop) { bestTop = r.top; best = it; }
    });
    if (measurable) setActive(best.getAttribute('data-lp-key'));
  }

  function bindSpy() {
    if (spyBound) return;
    spyBound = true;
    window.addEventListener('scroll', function () {
      if (spyScheduled) return;
      spyScheduled = true;
      var raf = window.requestAnimationFrame || function (f) { return setTimeout(f, 16); };
      raf(function () { spyScheduled = false; spy(); });
    }, { passive: true });
  }

  function renderStatus() {
    if (!statusBtn) return;
    var ms = LP.msgState();
    var cls = ms.err ? 'err' : (ms.warn ? 'warn' : 'ok');
    var txt = ms.err ? (ms.err + ' 错误') : (ms.warn ? (ms.warn + ' 提醒') : '可运行');
    statusBtn.className = 'lp-rail-status ' + cls;
    statusBtn.innerHTML = '<i aria-hidden="true"></i><span>' + esc(txt) + '</span>';
    statusBtn.title = ms.err ? ('校验有 ' + ms.err + ' 个错误（点击查看）')
      : ms.warn ? ('有 ' + ms.warn + ' 条提醒（点击查看）')
      : '校验通过';
  }

  function markScenario() {
    if (!scenGroup) return;
    var cur = LP.body.dataset.scenario;
    Array.prototype.forEach.call(scenGroup.children, function (b) {
      var on = b.getAttribute('data-lp-scenario') === cur;
      b.classList.toggle('on', on);
      if (on) b.setAttribute('aria-current', 'true');
      else b.removeAttribute('aria-current');
    });
  }

  LP.register({
    rebuild: function () {
      ensureRail();
      buildSecs();
      markScenario();
      renderStatus();
    },
    patch: function () {
      markScenario();
      renderStatus();
    }
  });
  LP.start();
})();
