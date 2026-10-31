/* ══════════════════════════════════════════════════════════════════════════
 * 变体 B「Bento 面板」—— 不对称磁贴 + 常驻底部操作条
 *
 * 使用方式的变化：一屏纵览所有分区（磁贴），底部常驻「校验状态 / 复制 / 生成」，
 * 不再需要滚到右栏找按钮。
 * 只在 body 末尾加一条固定操作条，不改动 #configCol / #outCol 的任何节点。
 * ══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';
  var LP = window.__LP__;
  if (!LP || LP.arch !== 'bento') return;

  var $ = function (id) { return document.getElementById(id); };
  var bar = null, statusBtn = null, flashTimer = null;

  function ensureBar() {
    if (bar && document.body.contains(bar)) return;
    bar = document.createElement('div');
    bar.id = 'lpBentoBar';
    bar.setAttribute('role', 'toolbar');
    bar.setAttribute('aria-label', '操作');
    bar.innerHTML =
      '<button type="button" class="lp-bar-status" id="lpBarStatus" title="查看校验结果">' +
        '<i aria-hidden="true"></i><span>—</span>' +
      '</button>' +
      '<div class="lp-bar-gap" role="presentation"></div>' +
      '<button type="button" class="lp-bar-btn" id="lpBarCopy">复制当前</button>' +
      '<button type="button" class="lp-bar-btn primary" id="lpBarGen">生成命令</button>';
    document.body.appendChild(bar);
    statusBtn = bar.querySelector('#lpBarStatus');
    statusBtn.addEventListener('click', function () {
      var m = $('msgs');
      if (m && m.scrollIntoView) m.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
    bar.querySelector('#lpBarCopy').addEventListener('click', copyNow);
    bar.querySelector('#lpBarGen').addEventListener('click', generate);
  }

  function scriptCard() {
    var c = $('cmd');
    return (c && c.closest) ? c.closest('.card') : null;
  }

  function flash() {
    var card = scriptCard();
    if (!card) return;
    card.classList.add('lp-flash');
    if (flashTimer) clearTimeout(flashTimer);
    flashTimer = setTimeout(function () { card.classList.remove('lp-flash'); }, 1200);
  }

  function copyNow() {
    var b = $('copyBtn');
    if (!b) return;
    if (b.disabled) { LP.toast('先修正校验错误再复制'); return; }
    b.click();
    flash();
  }

  function generate() {
    try { if (window.BUILDER && window.BUILDER.refresh) window.BUILDER.refresh(); } catch (e) { /* noop */ }
    if (LP.layout.classList.contains('rail-collapsed')) {
      try { window.BUILDER.toggleSidebar(); } catch (e) { /* noop */ }
    }
    var rail = $('sideRail') || $('outCol');
    if (rail && rail.scrollIntoView) {
      try { rail.scrollIntoView({ block: 'start', behavior: 'smooth' }); }
      catch (e) { rail.scrollIntoView(); }
    }
    flash();
  }

  function renderStatus() {
    if (!statusBtn) return;
    var ms = LP.msgState();
    var cls = ms.err ? 'err' : (ms.warn ? 'warn' : 'ok');
    var txt = ms.err ? (ms.err + ' 个错误') : (ms.warn ? (ms.warn + ' 条提醒') : '校验通过');
    statusBtn.className = 'lp-bar-status ' + cls;
    statusBtn.innerHTML = '<i aria-hidden="true"></i><span>' + txt + '</span>';
  }

  LP.register({
    rebuild: function () { ensureBar(); renderStatus(); },
    patch: function () { renderStatus(); }
  });
  LP.start();
})();
