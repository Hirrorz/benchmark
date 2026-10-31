/* ══════════════════════════════════════════════════════════════════════════
 * 布局包公共运行时（注入在每个变体的 </body> 之前、主脚本之后执行）
 *
 *   - 主脚本末尾已同步跑过 init()/bind()，所以这里能直接用 window.BUILDER /
 *     window.setEntry / window.applyTheme / window.showToast。
 *   - 只加壳、不搬卡：本文件不创建/移动任何原有节点，只读它们的状态。
 *   - 状态签名 = scenario | uiMode | 可见卡 | layout 类 | 校验消息数；
 *     签名变了才 rebuild，否则只 patch → 观察器不会自激。
 *   - window.__LP__ 暴露给测试与 archetype 文件。
 * ══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';
  var ARCH = window.__LP_ARCHETYPE__;
  if (!ARCH || window.__LP__) return;

  var $ = function (id) { return document.getElementById(id); };
  var body = document.body;
  var layout = $('appLayout');
  var cfg = $('configCol');
  if (!layout || !cfg) return;

  body.classList.add('lp-' + ARCH);

  /* 变体身份主题：Studio 深色极光 / Bento 浅色糖果；之后用户可用 #themeBtn 切另一档 */
  try {
    var want = (ARCH === 'studio') ? 'dark' : 'light';
    if (typeof window.applyTheme === 'function') window.applyTheme(want);
    else document.documentElement.setAttribute('data-theme', want);
  } catch (e) { /* noop */ }

  function isVisible(card) {
    if (!card || !card.classList) return false;
    if (card.classList.contains('hidden')) return false;
    if (body.dataset.uiMode === 'simple' && card.classList.contains('adv-only')) return false;
    return true;
  }
  function cards() {
    return Array.prototype.filter.call(cfg.children, function (el) {
      return el.classList && el.classList.contains('card');
    });
  }
  function visibleCards() { return cards().filter(isVisible); }
  function cardKey(c) { return (c && c.id) || 'common'; }

  /* 卡片标题：取 .cardhead h2 的文本，但跳过徽标（badge）里的动态数字 */
  function cardTitle(card) {
    var h = card.querySelector('.cardhead h2');
    if (!h) return cardKey(card);
    var out = '';
    Array.prototype.forEach.call(h.childNodes, function (n) {
      if (n.nodeType === 3) out += n.textContent;
      else if (n.nodeType === 1 && !(n.classList && n.classList.contains('badge')))
        out += n.textContent;
    });
    out = (out || h.textContent || '').replace(/\s+/g, ' ').trim();
    return out || cardKey(card);
  }

  /* 校验状态（#msgs 由 refresh() 重渲染，这里只数类名） */
  function msgState() {
    var m = $('msgs'), err = 0, warn = 0, okN = 0;
    if (m) {
      err = m.querySelectorAll('.msg.err').length;
      warn = m.querySelectorAll('.msg.warn').length;
      okN = m.querySelectorAll('.msg.ok').length;
    }
    return { err: err, warn: warn, ok: okN };
  }

  var PRIMARY = {
    points: 'pointsCard', resident: 'residentCard', launch: 'launchCard',
    hit: 'hitCard', real: 'realCard'
  };

  var hooks = null, sig = '', scheduled = false, syncing = false;

  function signature() {
    var ms = msgState();
    return [body.dataset.scenario || '', body.dataset.uiMode || '',
      visibleCards().map(cardKey).join(','), layout.className,
      ms.err + '/' + ms.warn + '/' + ms.ok].join('|');
  }

  function sync(force) {
    if (syncing) return;
    syncing = true;
    try {
      var s = signature();
      if (force || s !== sig) {
        sig = s;
        if (hooks && hooks.rebuild) hooks.rebuild();
      } else if (hooks && hooks.patch) {
        hooks.patch();
      }
    } finally { syncing = false; }
  }

  function schedule() {
    if (scheduled) return;
    scheduled = true;
    var raf = window.requestAnimationFrame || function (f) { return setTimeout(f, 16); };
    raf(function () { scheduled = false; try { sync(false); } catch (e) { /* noop */ } });
  }

  function observe(target, opts) {
    if (typeof MutationObserver === 'undefined' || !target) return;
    try { new MutationObserver(schedule).observe(target, opts); } catch (e) { /* noop */ }
  }

  /* 打印：展开所有 <details>（壳层由 shared.css 的 @media print 隐藏），打印后恢复 */
  var _printOpened = [];
  function _beforePrint() {
    _printOpened = [];
    Array.prototype.forEach.call(document.querySelectorAll('details:not([open])'),
      function (d) { d.setAttribute('open', ''); _printOpened.push(d); });
    body.classList.add('lp-printing');
  }
  function _afterPrint() {
    body.classList.remove('lp-printing');
    _printOpened.forEach(function (d) { d.removeAttribute('open'); });
    _printOpened = [];
  }
  window.addEventListener('beforeprint', _beforePrint);
  window.addEventListener('afterprint', _afterPrint);

  window.__LP__ = {
    arch: ARCH, body: body, layout: layout, cfg: cfg,
    cards: cards, visibleCards: visibleCards, isVisible: isVisible,
    cardTitle: cardTitle, cardKey: cardKey, primary: PRIMARY,
    msgState: msgState, sync: sync, schedule: schedule,
    register: function (h) { hooks = h; },
    start: function () {
      sync(true);
      observe(body, { attributes: true, attributeFilter: ['data-scenario', 'data-ui-mode'] });
      observe(cfg, { subtree: true, attributes: true, attributeFilter: ['class'] });
      observe(layout, { attributes: true, attributeFilter: ['class'] });
      observe($('msgs'), { childList: true, subtree: true, characterData: true });
    },
    toast: function (m) { try { if (window.showToast) window.showToast(m); } catch (e) { /* noop */ } }
  };
})();
