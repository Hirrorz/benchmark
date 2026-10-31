/* 布局包变体的结构 + 行为测试（jsdom）
 *
 *   JSDOM_PATH=/tmp/jsdom_env/node_modules/jsdom node tools/test_layout_variants.js
 *
 * 覆盖：
 *   1) 源码结构：布局包标记成对、只有一个 style/script pack、位于 </body> 前
 *   2) DOM 不变量：无重复 id、基础 id 全集、#configCol 10 张直接子卡、#outCol 首卡不变、
 *      #outBand 通栏在 #appLayout 首位且含「生成的脚本」卡
 *   3) 运行时：body.lp-*、主题初值、场景切换后导航/底栏同步、校验状态、侧栏折叠仍生效
 */
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require(process.env.JSDOM_PATH || 'jsdom');

const HERE = __dirname;
const BASE = path.join(HERE, 'bench_command_builder.html');
const VARIANTS = {
  studio: path.join(HERE, 'bench_command_builder.layout_studio.html'),
  bento: path.join(HERE, 'bench_command_builder.layout_bento.html'),
};

let pass = 0, fail = 0;
const errors = [];
function ok(cond, msg) { if (cond) pass++; else { fail++; errors.push('FAIL: ' + msg); } }
function has(s, sub, msg) { ok(String(s).includes(sub), msg + ' (missing: ' + sub + ')'); }

function load(file) {
  const html = fs.readFileSync(file, 'utf8');
  const vc = new VirtualConsole();
  vc.on('jsdomError', e => { fail++; errors.push(`JSDOM ERROR(${path.basename(file)}): ${e.message}`); });
  vc.on('error', (...a) => { fail++; errors.push(`CONSOLE ERROR(${path.basename(file)}): ` + a.join(' ')); });
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://ds41.test/tools/bench_command_builder.html',
    virtualConsole: vc,
    beforeParse(window) {
      Object.defineProperty(window.navigator, 'clipboard', {
        value: { writeText: async t => { window.__copied = t; } }, configurable: true,
      });
      window.URL.createObjectURL = () => 'blob:mock';
      window.URL.revokeObjectURL = () => {};
      window.document.execCommand = () => true;
    },
  });
  dom.window.HTMLAnchorElement.prototype.click = function () { this.__clicked = true; };
  return dom;
}

const baseHtml = fs.readFileSync(BASE, 'utf8');
const baseDom = new JSDOM(baseHtml);                 // 不跑脚本，只取 id 全集
const baseDoc = baseDom.window.document;
const baseIds = new Set([...baseDoc.querySelectorAll('[id]')].map(e => e.id));
const baseCfgCards = [...baseDoc.getElementById('configCol').children]
  .filter(e => e.classList.contains('card'));
const baseOutCards = [...baseDoc.getElementById('outCol').children]
  .filter(e => e.classList.contains('card'));
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  for (const [arch, file] of Object.entries(VARIANTS)) {
    const html = fs.readFileSync(file, 'utf8');

    /* ── 1) 源码结构 ── */
    const cs = `<!-- LP-PACK:${arch}:CSS:START -->`, ce = `<!-- LP-PACK:${arch}:CSS:END -->`;
    const js = `<!-- LP-PACK:${arch}:JS:START -->`, je = `<!-- LP-PACK:${arch}:JS:END -->`;
    ok(html.includes(cs) && html.includes(ce), `${arch}: CSS 标记成对`);
    ok(html.includes(js) && html.includes(je), `${arch}: JS 标记成对`);
    ok(html.indexOf(cs) < html.indexOf(ce) && html.indexOf(ce) < html.indexOf('</head>'),
       `${arch}: 样式包在 </head> 之前`);
    ok(html.indexOf(js) > html.indexOf('</head>') && html.indexOf(je) < html.lastIndexOf('</body>'),
       `${arch}: 脚本包在 </body> 之前、且在 head 之后`);
    ok((html.match(/id="lp-pack-style"/g) || []).length === 1 &&
       (html.match(/id="lp-pack-script"/g) || []).length === 1,
       `${arch}: 恰好一个 style + 一个 script pack`);
    ok(!/LP-PACK:[a-z0-9_-]+:(CSS|JS):START/.test(html.slice(html.indexOf(je) + je.length)),
       `${arch}: JS 包之后没有残留标记`);
    ok(!/LP-PACK:[a-z0-9_-]+:START -->/.test(html), `${arch}: 没有旧版单块标记残留`);
    has(html, `window.__LP_ARCHETYPE__ = "${arch}"`, `${arch}: archetype 常量`);
    ok(/@media print/.test(html), `${arch}: 打印样式存在`);
    ok(/beforeprint/.test(html) && /lp-printing/.test(html),
       `${arch}: 打印前展开钩子存在`);

    /* ── 2) DOM 不变量 ── */
    const dom = load(file);
    const w = dom.window, d = w.document;
    await sleep(60);
    const ids = [...d.querySelectorAll('[id]')].map(e => e.id);
    const dup = ids.filter((v, i) => ids.indexOf(v) !== i);
    ok(dup.length === 0, `${arch}: 无重复 id（${dup.slice(0, 3).join(',')}）`);
    const missing = [...baseIds].filter(id => !d.getElementById(id));
    ok(missing.length === 0, `${arch}: 基础 id 全集存活（缺 ${missing.slice(0, 3).join(',')}）`);

    const cfgCards = [...d.getElementById('configCol').children]
      .filter(e => e.classList.contains('card'));
    ok(cfgCards.length === baseCfgCards.length,
       `${arch}: #configCol 仍有 ${baseCfgCards.length} 张直接子卡（实际 ${cfgCards.length}）`);
    ok(cfgCards[0] === d.getElementById('configCol').firstElementChild,
       `${arch}: 通用卡仍是 #configCol 首个子节点`);
    const outCards = [...d.getElementById('outCol').children]
      .filter(e => e.classList.contains('card'));
    ok(outCards.length === baseOutCards.length,
       `${arch}: #outCol 仍有 ${baseOutCards.length} 张卡（实际 ${outCards.length}）`);
    ok(outCards[0].contains(d.getElementById('chips')), `${arch}: chips 卡仍是 #outCol 首卡`);
    const band = d.getElementById('outBand');
    ok(!!band, `${arch}: 顶部脚本通栏存在`);
    ok(band.parentElement === d.getElementById('appLayout'),
       `${arch}: #outBand 是 #appLayout 的直接子节点（脚本置顶）`);
    if (arch === 'bento')
      ok(band === d.getElementById('appLayout').firstElementChild,
         'bento: #outBand 是首个子节点（脚本置顶）');
    if (arch === 'studio')
      ok(band.previousElementSibling === d.getElementById('lpStudioRail'),
         'studio: #outBand 紧跟左导航（同一行）');
    ok(band.contains(d.getElementById('scriptCard')) && band.contains(d.getElementById('cmd')),
       `${arch}: 生成的脚本卡在通栏里`);
    ok(!d.getElementById('outCol').contains(d.getElementById('scriptCard')),
       `${arch}: 脚本卡不再挤在右侧栏`);
    ok(band.contains(d.getElementById('tabs')) && band.contains(d.getElementById('copyBtn')),
       `${arch}: tabs / 复制按钮随卡一起在通栏`);
    const bottom = d.getElementById('outBottom');
    ok(!!bottom && bottom.parentElement === d.getElementById('appLayout'),
       `${arch}: 校验/摘要通栏是 #appLayout 直接子节点`);
    ok(bottom === d.getElementById('appLayout').lastElementChild,
       `${arch}: 校验/摘要在页面最底部`);
    ok(bottom.contains(d.getElementById('msgs')) && bottom.contains(d.getElementById('summary')),
       `${arch}: 校验与摘要都在底部通栏`);
    ok(!d.getElementById('msgsCard').hasAttribute('open') &&
       !d.getElementById('summaryCard').hasAttribute('open'),
       `${arch}: 两个折叠面板默认收起`);

    /* ── 3) 运行时 ── */
    ok(d.body.classList.contains('lp-' + arch), `${arch}: body 有 lp-${arch} 类`);
    const wantTheme = arch === 'studio' ? 'dark' : 'light';
    ok(d.documentElement.getAttribute('data-theme') === wantTheme,
       `${arch}: 初值主题 = ${wantTheme}`);
    ok(w.__LP__ && w.__LP__.arch === arch, `${arch}: __LP__ 已挂载`);
    ok(typeof w.setEntry === 'function' && typeof w.refresh === 'function',
       `${arch}: 兼容层未被破坏`);

    const expectedVisible = () => [...d.getElementById('configCol').children].filter(e =>
      e.classList.contains('card') &&
      !e.classList.contains('hidden') &&
      !(d.body.dataset.uiMode === 'simple' && e.classList.contains('adv-only'))).length;

    if (arch === 'studio') {
      const rail = d.getElementById('lpStudioRail');
      ok(!!rail, 'studio: 左导航存在');
      ok(rail.parentElement === d.getElementById('appLayout'), 'studio: 导航是 #appLayout 子节点');
      const secs = () => d.querySelectorAll('#lpRailSecs .lp-rail-sec').length;
      ok(secs() === expectedVisible(), `studio: 分区条目=可见卡（${secs()} vs ${expectedVisible()}）`);
      w.setEntry('resident');
      await sleep(80);
      ok(secs() === expectedVisible(), `studio: 换场景后条目同步（${secs()} vs ${expectedVisible()}）`);
      const on = d.querySelector('#lpRailScen .lp-rail-scen.on');
      ok(on && on.getAttribute('data-lp-scenario') === 'resident', 'studio: 场景按钮高亮跟随');
      ok(d.querySelectorAll('#lpRailScen .lp-rail-scen').length === 5, 'studio: 5 个场景按钮');
      const st = d.getElementById('lpRailStatus');
      ok(st && /ok|warn|err/.test(st.className), 'studio: 状态按钮已渲染');
    }

    if (arch === 'bento') {
      ok(!!d.getElementById('lpBentoBar'), 'bento: 底部操作条存在');
      const st = d.getElementById('lpBarStatus');
      ok(st && /ok|warn|err/.test(st.className), 'bento: 状态已渲染');
      const cards = d.getElementById('cards');
      const prev = cards.value;
      cards.value = '';
      cards.dispatchEvent(new w.Event('input', { bubbles: true }));
      await sleep(80);
      ok(st.className.includes('err'), 'bento: 校验错误 → 状态点变红');
      ok(st.textContent.includes('错误'), 'bento: 状态文案含「错误」');
      cards.value = prev;
      cards.dispatchEvent(new w.Event('input', { bubbles: true }));
      await sleep(80);
      ok(!st.className.includes('err'), 'bento: 修好后状态恢复');
      const layout = d.getElementById('appLayout');
      if (layout.classList.contains('rail-collapsed')) w.BUILDER.toggleSidebar();
      w.BUILDER.toggleSidebar();
      ok(layout.classList.contains('rail-collapsed'), 'bento: 侧栏折叠仍生效');
      d.getElementById('lpBarGen').click();
      await sleep(30);
      ok(!layout.classList.contains('rail-collapsed'), 'bento: 生成命令会展开折叠的右栏');
    }

    /* 打印：展开折叠面板，打印后恢复 */
    const qs = d.getElementById('quickStart');
    if (qs) {
      qs.removeAttribute('open');
      w.dispatchEvent(new w.Event('beforeprint'));
      ok(qs.hasAttribute('open'), `${arch}: 打印前展开 details`);
      ok(d.body.classList.contains('lp-printing'), `${arch}: lp-printing 类`);
      w.dispatchEvent(new w.Event('afterprint'));
      ok(!qs.hasAttribute('open'), `${arch}: 打印后恢复折叠`);
      ok(!d.body.classList.contains('lp-printing'), `${arch}: lp-printing 类移除`);
    }

    /* 侧栏折叠 / 主题开关在两套变体里都还能用 */
    const layout = d.getElementById('appLayout');
    const before = layout.classList.contains('rail-collapsed');
    w.BUILDER.toggleSidebar();
    ok(layout.classList.contains('rail-collapsed') !== before, `${arch}: 折叠开关生效`);
    w.BUILDER.toggleSidebar();
    const th0 = d.documentElement.getAttribute('data-theme');
    d.getElementById('themeBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
    ok(d.documentElement.getAttribute('data-theme') !== th0, `${arch}: 主题按钮仍可切换`);

    /* 场景导航节点身份不变（逐节点绑定的监听不能被重建） */
    const navBefore = [...d.querySelectorAll('#scenarioNav [data-nav]')];
    w.setEntry('points');
    await sleep(40);
    ok(navBefore.every(b => b.isConnected), `${arch}: #scenarioNav 节点未被重建`);
    ok(d.querySelectorAll('#scenarioNav [data-nav]').length === 4, `${arch}: 场景导航 4 项（hit 已并入多点）`);

    dom.window.close();
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  if (errors.length) console.log(errors.join('\n'));
  process.exit(fail ? 1 : 0);
})().catch(e => { console.log('FATAL: ' + (e && e.stack || e)); process.exit(2); });
