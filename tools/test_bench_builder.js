/* bench_command_builder.html v2 的功能测试（jsdom）
 *
 *   npm i jsdom   或   JSDOM_PATH=/path/to/jsdom node tools/test_bench_builder.js
 *
 * 覆盖：
 *   0) 初始化 / 元素完整性 / 兼容层
 *   1) 常驻 ep8 golden（server / warmup / run / status / stop / tail）
 *   2) 请求队列（增删、顺序、空队列阻断）
 *   3) 模式（ep8 / auto / tp8 未验证 / tp8-dual / host cache）
 *   4) 校验（卡号、阶梯、池容量、重复卡、非法 env）
 *   5) dry-run（start/warmup/run，收尾注释）
 *   6) 一键脚本（nohup / socket 等待 / trap）
 *   7) 单跑 launch（smoke/packed/decode/prompt_decode，作用域隔离）
 *   8) 单跑 hit（perf/plain/dry-run/no-bar/非法命中率）
 *   9) 单跑 real（prefill/parity/no-precision/dry-run/互斥）
 *  10) 高级（serve env、suboptimal、客户端开关、内部项不暴露）
 *  11) 导入 / 导出（JSON round-trip、URL hash、坏 JSON）
 *  12) 预设重置 / 高亮清除
 *  13) 多点（--points）：自定义 / 导入 / replay_tail / moemode / 文件·内联传递 / 两条命令
 *  14) 使用说明 / 复制 / 下载 / 分享
 *  15) 布局：右侧场景栏 / 折叠持久化 / 多点优先 / 运行轮次三选一
 *
 * 返回码：0=全过；1=有失败；2=测试本身异常。
 */
/* Functional tests for tools/bench_command_builder.html (jsdom). */
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { JSDOM, VirtualConsole } = require(process.env.JSDOM_PATH || 'jsdom');

// BENCH_HTML：跑指定变体（布局包 CLI 用）；默认仍是主构建器
const HTML = process.env.BENCH_HTML
  ? path.resolve(process.env.BENCH_HTML)
  : path.join(__dirname, 'bench_command_builder.html');
const html = fs.readFileSync(HTML, 'utf8');
const scriptText = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

let pass = 0, fail = 0;
const errors = [];
function ok(cond, msg) {
  if (cond) { pass++; }
  else { fail++; errors.push('FAIL: ' + msg); }
}
function has(s, sub, msg) { ok(String(s).includes(sub), msg + ' (missing: ' + sub + ')'); }
function lacks(s, sub, msg) { ok(!String(s).includes(sub), msg + ' (unexpected: ' + sub + ')'); }

const vc = new VirtualConsole();
vc.on('jsdomError', e => { errors.push('JSDOM ERROR: ' + e.message); fail++; });
vc.on('error', (...a) => { errors.push('CONSOLE ERROR: ' + a.join(' ')); fail++; });

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  // 用 https 源：file:// 是不透明源，jsdom 的 localStorage 会抛 SecurityError，
  // 而侧栏折叠 / 场景记忆都要测 localStorage。功能本身仍然离线可用。
  url: 'https://ds41.test/tools/bench_command_builder.html',
  virtualConsole: vc,
  beforeParse(window) {
    Object.defineProperty(window.navigator, 'clipboard', {
      value: { writeText: async t => { window.__copied = t; } },
      configurable: true,
    });
    window.URL.createObjectURL = () => 'blob:mock';
    window.URL.revokeObjectURL = () => {};
    window.document.execCommand = () => true;
  },
});
const w = dom.window, d = w.document;
w.HTMLAnchorElement.prototype.click = function () { w.__lastDownload = this.download; };

/* ── helpers ── */
function $(id) { return d.getElementById(id); }
function setVal(id, v) { $(id).value = v; }
function setChk(id, v) { $(id).checked = !!v; }
function fire(el, type) { el.dispatchEvent(new w.Event(type || 'input', { bubbles: true })); }
function setAndFire(id, v, type) { setVal(id, v); fire($(id), type || 'input'); }
function checkAndFire(id, v) { setChk(id, v); fire($(id), 'change'); }
function cmd() { w.refresh(); return $('cmd').textContent; }
function msgs() { w.refresh(); return $('msgs').textContent; }
function E() { return w.BUILDER.collectWarnings().E; }
function W() { return w.BUILDER.collectWarnings().W; }
function resetAll() { w.BUILDER.applyAllDefaults(); w.refresh(); }
function resident() { resetAll(); w.BUILDER.PRESETS.resident(); w.refresh(); }
function queueRows() { return [...d.querySelectorAll('#resQueue .qrow')]; }
function setPrompt(i, text) {
  const areas = d.querySelectorAll('#resQueue .q-prompt');
  areas[i].value = text;
  fire(areas[i]);
}
function selectTab(id) {
  w.refresh();
  const b = d.querySelector(`#tabs .tab[data-tab="${id}"]`);
  if (!b) return '';
  b.dispatchEvent(new w.Event('click', { bubbles: true }));
  return $('cmd').textContent;
}
function tabIds() { return [...d.querySelectorAll('#tabs .tab')].map(x => x.dataset.tab); }
function count(s, sub) { return String(s).split(sub).length - 1; }

setTimeout(() => {
try {
  /* ══════════ 0) 初始化 / 元素完整性 / 兼容层 ══════════ */
  ok(typeof w.setEntry === 'function' && typeof w.refresh === 'function',
     'compat: setEntry/refresh exposed');
  ok(typeof w.buildResidentCommand === 'function', 'compat: buildResidentCommand exposed');
  ok(Array.isArray(w.CONFIGS) && w.CONFIGS.length >= 13, 'compat: CONFIGS exposed');
  ok(w.PRESETS && typeof w.PRESETS.resident === 'function', 'compat: PRESETS exposed');
  ok(w.BUILDER && typeof w.BUILDER.getState === 'function', 'compat: BUILDER exposed');
  ok($('config').options.length === w.eval('CONFIGS.length'), 'config select populated');

  const usedIds = new Set();
  for (const re of [/\$\('([^']+)'\)/g, /getElementById\('([^']+)'\)/g]) {
    for (const m of scriptText.matchAll(re)) usedIds.add(m[1]);
  }
  const missing = [...usedIds].filter(id => !$(id));
  ok(missing.length === 0, 'all static IDs exist: ' + missing.join(','));

  ok(d.body.dataset.uiMode === 'simple', 'init: simple mode');
  ok(d.body.dataset.scenario === 'points', 'init: points scenario (points-first)');
  ok(!$('pointsCard').classList.contains('hidden'), 'init: points card visible');
  has($('pointsCard').textContent, '合成权重模块（性能）', 'modules: points title');
  has($('realCard').textContent, '真权重模块（精度）', 'modules: real title');
  has($('realCard').querySelector('.hint').textContent, '悬停',
     'modules: real card advertises hover help');
  // 环境自检是**一次性**的：单独陈列，不进生成脚本
  lacks(selectTab('cmd'), 'tools/selfcheck.py', 'selfcheck: not embedded in generated scripts');
  const _script0 = selectTab('cmd');
  has(_script0, '可整段粘贴', 'scripts: paste-safe note');
  ok(/\(\nset -euo pipefail/.test(_script0), 'scripts: wrapped in subshell');
  ok(_script0.trimEnd().endsWith(')'), 'scripts: subshell closed');
  has($('quickStart').textContent, '一次性环境自检', 'selfcheck: shown separately in quick start');
  has($('quickStart').textContent, 'tools/selfcheck.py --card', 'selfcheck: command shown');
  ok($('realParity').checked, 'modules: real --parity default on');
  // 性能口径跳过开关（常配 1 卡：不跑 MoE / engram，也不加载对应权重）
  ok($('skipMoe') && !$('skipMoe').checked, 'skip: MoE switch present, default off');
  ok($('skipEngram') && !$('skipEngram').checked, 'skip: engram switch present, default off');
  has($('skipMoe').dataset.help, '性能', 'skip: MoE switch explains perf-only');
  has($('skipEngram').dataset.help, 'engram', 'skip: engram switch explains what is skipped');
  ok(!!$('qsSingleCard'), 'skip: quick-start single-card button present');
  const _groups = [...d.querySelectorAll('#scenarioNav .side-group')].map(x => x.textContent.trim());
  ok(_groups.includes('合成权重 · 性能') && _groups.includes('真权重 · 精度'),
     'modules: nav has two module groups');
  ok(!d.querySelector('#scenarioNav [data-nav="hit"]'), 'modules: hit not a separate nav entry');
  // 悬停说明：字段标题有 data-help，控件有原生 title
  const _plField = $('pointsDelivery').closest('.field');
  ok(!!_plField && !!_plField.querySelector('.k').dataset.help,
     'help: field title carries data-help');
  has(_plField.querySelector('.k').dataset.help, 'file=', 'help: option meanings merged');
  has($('pointsDelivery').title, 'file=', 'help: control has native title');
  has($('pointsMoemode').title, 'ascendc', 'help: moemode options explained');
  has($('pdMode').title, 'prefill', 'help: P→D off (reuse prefill weights) explained');
  has($('alopMindstudio').title, 'trace_view', 'help: MindStudio trace vs built-in ALOP explained');
  ok(!$('profOnly') && !$('noprofOnly'), 'launch: two-leg toggles not shown (launch has one leg)');
  has($('resWarmOnStart').title, '服务端启动', 'help: 启动后自动预热 explained');
  const _rowL = d.querySelector('#pointsTbody .pt-l');
  ok(!!_rowL && /128/.test(_rowL.title), 'help: points L input explains 128 alignment');
  ok($('residentCard').classList.contains('hidden'), 'init: resident card hidden');
  ok($('queueCard').classList.contains('hidden'), 'init: queue card hidden');
  ok($('pointsRounds').value === 'both', 'init: points rounds default both');
  const navOrder = [...d.querySelectorAll('#moreScenarios [data-nav]')].map(x => x.dataset.nav);
  ok(navOrder.join(',') === 'points,launch,real,resident',
     'init: sidebar order points first (' + navOrder.join(',') + ')');
  ok(d.querySelectorAll('#moreScenarios .side-group').length === 2,
     'nav: two weight modules (合成/真权重)');
  ok(!d.querySelector('#scenarioNav [data-nav="hit"]'),
     'nav: hit merged into points (no separate entry)');
  has(d.querySelector('#moreScenarios').textContent, '合成权重', 'nav: synthetic group label');
  has(d.querySelector('#moreScenarios').textContent, '真权重', 'nav: real group label');
  const pnav = d.querySelector('#scenarioNav [data-nav="points"]');
  ok(pnav.classList.contains('on'), 'init: points tab active');
  ok(pnav.getAttribute('aria-pressed') === 'true', 'init: points tab aria-pressed');
  ok(pnav.querySelector('.side-badge') &&
     pnav.querySelector('.side-badge').textContent.includes('推荐'),
     'init: points tab has 推荐 badge');
  ok(/多点/.test($('moreCurrent').textContent), 'init: rail title shows points');
  ok($('sidebarToggle').getAttribute('aria-expanded') === 'true',
     'init: sidebar expanded by default');
  ok(!$('appLayout').classList.contains('rail-collapsed'),
     'init: rail not collapsed');
  has(html, 'body[data-scenario="points"] #configCol > #pointsCard{order:-1}',
     'init: points card promoted to top by CSS');
  let c = cmd();
  has(c, 'run_hit.sh', 'init: points command rendered');
  has(c, '--points', 'init: points delivery rendered');
  ok(!E().length, 'init: no blocking errors');

  // 场景切换：右侧竖排 tab 仍然复用原有 hidden/show 逻辑
  pnav.dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(d.body.dataset.scenario === 'points', 'nav: points tab keeps points');
  const rnav = d.querySelector('#scenarioNav [data-nav="resident"]');
  rnav.dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(d.body.dataset.scenario === 'resident', 'nav: resident tab switches scenario');
  ok(!$('residentCard').classList.contains('hidden') &&
     $('pointsCard').classList.contains('hidden'), 'nav: resident card shown');
  ok(rnav.classList.contains('on') && !pnav.classList.contains('on'),
     'nav: active indicator follows tab');
  ok(rnav.getAttribute('aria-pressed') === 'true' &&
     pnav.getAttribute('aria-pressed') === 'false', 'nav: aria-pressed follows tab');
  ok(w.localStorage.getItem('ds41-bench-builder-scenario') === 'resident',
     'nav: scenario persisted to localStorage');

  // 恢复 points 场景，再按原有常驻 golden 检查渲染
  w.setEntry('points');
  w.refresh();
  w.setEntry('resident');
  w.refresh();
  ok($('resMode').value === 'ep8', 'init: ep8 default');
  ok(!$('resHostCache').checked, 'init: host cache off');
  ok($('resSharedTmp').checked, 'init: shared tmp on');
  ok(queueRows().length === 1, 'init: one queue row');
  ok(!d.querySelector('#presets button[data-preset="packed_on"]'),
     'init: packed preset removed');
  c = cmd();
  has(c, 'bench_server.py start', 'init: server command rendered');
  has(c, 'bench_cli.py warmup', 'init: warmup rendered');
  has(c, 'bench_cli.py run', 'init: run rendered');
  has(c, 'bench_cli.py status', 'init: status rendered');
  has(c, 'bench_cli.py stop', 'init: stop rendered');
  ok(typeof w.buildCommand === 'function' && w.buildCommand() === $('cmd').textContent,
     'compat: buildCommand returns current tab');
  ok(w.BUILDER.buildScripts().tabs.length >= 4, 'compat: buildScripts tabs');

  /* ══════════ 0.5) 右侧场景栏：折叠 / 持久化 / 窄屏 ══════════ */
  const railBtn = $('sidebarToggle');
  const layoutEl = $('appLayout');
  ok(typeof w.BUILDER.toggleSidebar === 'function', 'sidebar: toggle exposed');
  const _th0 = d.documentElement.getAttribute('data-theme');
  $('themeBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(d.documentElement.getAttribute('data-theme') !== _th0, 'theme: toggle works (base)');
  $('themeBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(railBtn.getAttribute('aria-controls') === 'sideBody', 'sidebar: aria-controls');
  w.BUILDER.toggleSidebar();
  ok(layoutEl.classList.contains('rail-collapsed'), 'sidebar: collapse class');
  ok(railBtn.getAttribute('aria-expanded') === 'false', 'sidebar: aria-expanded false');
  ok(railBtn.getAttribute('aria-label') === '展开侧栏', 'sidebar: collapsed label');
  ok(w.localStorage.getItem('ds41-bench-builder-sidebar') === 'collapsed',
     'sidebar: collapsed persisted');
  w.BUILDER.toggleSidebar();
  ok(!layoutEl.classList.contains('rail-collapsed'), 'sidebar: expands again');
  ok(railBtn.getAttribute('aria-expanded') === 'true', 'sidebar: aria-expanded true');
  ok(w.localStorage.getItem('ds41-bench-builder-sidebar') === 'expanded',
     'sidebar: expanded persisted');

  // <1100px 时侧栏变成顶部横条，折叠态不生效（CSS 同样限定桌面宽度）
  const oldMM = w.matchMedia;
  w.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
  w.BUILDER.toggleSidebar();
  ok(!layoutEl.classList.contains('rail-collapsed'), 'sidebar: narrow ignores collapse');
  ok(railBtn.getAttribute('aria-expanded') === 'true',
     'sidebar: narrow aria stays expanded');
  w.BUILDER.toggleSidebar();
  if (oldMM === undefined) delete w.matchMedia; else w.matchMedia = oldMM;
  ok(w.localStorage.getItem('ds41-bench-builder-sidebar') === 'expanded',
     'sidebar: narrow toggle did not persist collapsed');
  has(html, '.side-rail{display:contents}', 'sidebar: narrow rail becomes top bar');
  has(html, '@media (min-width:1600px)', 'layout: extra-wide two-column breakpoint');
  has(html, 'max-width:1920px', 'layout: wide max width');

  /* ══════════ 1) 常驻 ep8 golden ══════════ */
  resident();
  c = cmd();
  has(c, 'npu-smi info', 'ep8: air check');
  has(c, '--mode ep8', 'ep8: mode');
  has(c, '--host-cache off', 'ep8: host cache off');
  has(c, '--serve-l-max 1024', 'ep8: pool');
  has(c, '--lengths 16,32,64,128,256,512,1024', 'ep8: ladder');
  has(c, '--decode 32', 'ep8: decode');
  has(c, '--repeat 3', 'ep8: repeat');
  has(c, '--chat', 'ep8: chat');
  has(c, 'export DS41_BENCH_ROOT=', 'ep8: shared tmp export');
  has(c, 'tail -f', 'ep8: tail hint');
  lacks(c, 'moe_host_cache.py start', 'ep8: no host cache start');
  lacks(c, '--dry-run', 'ep8: no dry-run by default');
  const unRes = (c.match(/^unset .*$/m) || [''])[0];
  ok(/DS41M_PD_MERGE\b/.test(unRes) && /DS41M_TP_DECODE_ONLY\b/.test(unRes),
     'ep8: stale P→D env swept');
  ok(/DS41M_MODULE_RUNTIME\b/.test(unRes),
     'ep8: module runtime managed (stale =0 swept; runner default on)');
  ok(!E().length, 'ep8: no errors');
  ok(!W().length, 'ep8: no warnings');

  /* ══════════ 2) 请求队列 ══════════ */
  resident();
  setPrompt(0, '第一条');
  $('qAdd').dispatchEvent(new w.Event('click', { bubbles: true }));
  setPrompt(1, '第二条');
  $('qAdd').dispatchEvent(new w.Event('click', { bubbles: true }));
  setPrompt(2, '第三条');
  c = cmd();
  ok(queueRows().length === 3, 'queue: 3 rows');
  ok(count(c, 'bench_cli.py run') === 3, 'queue: 3 run commands');
  ok(c.indexOf("--prompt '第一条'") < c.indexOf("--prompt '第二条'") &&
     c.indexOf("--prompt '第二条'") < c.indexOf("--prompt '第三条'"),
     'queue: order preserved');
  has(c, 'run 1/3', 'queue: label 1/3');
  has(c, 'run 3/3', 'queue: label 3/3');
  queueRows()[1].querySelector('.qdel').dispatchEvent(new w.Event('click', { bubbles: true }));
  c = cmd();
  ok(queueRows().length === 2, 'queue: delete row');
  lacks(c, '第二条', 'queue: deleted prompt gone');
  has(c, 'run 2/2', 'queue: labels renumbered');
  w.BUILDER.setQueue([]); w.refresh();
  has(msgs(), '请求队列为空', 'queue: empty blocks');
  ok($('copyBtn').disabled, 'queue: empty disables copy');
  w.BUILDER.setQueue([{ prompt: '恢复', chat: true, thinking: 'chat', decode: 8, repeat: 1 }]);
  w.refresh();
  ok(!E().length, 'queue: restored clean');
  w.BUILDER.setQueue([{ prompt: '想一下', chat: true, thinking: 'thinking', decode: 4, repeat: 1 }]);
  has(cmd(), '--thinking thinking', 'queue: thinking passed');

  /* ══════════ 3) 模式 ══════════ */
  resident();
  w.BUILDER.PRESETS.resident_tp8();
  w.refresh();
  ok(!$('tp8Banner').classList.contains('hidden'), 'tp8: banner visible');
  ok($('resHostCache').checked, 'tp8: host cache auto-on');
  c = cmd();
  has(c, '--mode tp8-fidelity', 'tp8: mode');
  has(c, '--host-cache /tmp/ds41_moe_host_cache.sock', 'tp8: host cache arg');
  has(c, 'moe_host_cache.py start', 'tp8: daemon start');
  has(c, 'moe_host_cache.py stop --socket', 'tp8: daemon stop');
  has(c, '--progress-bar', 'tp8: progress bar');
  has(c, '强制 inc', 'tp8: script warning');
  has(msgs(), 'carry 仍是近似', 'tp8: validation warning');
  ok(!E().length, 'tp8: no blocking error');
  checkAndFire('resHostCache', false);
  has(msgs(), '需要 host cache', 'tp8 no-hc: blocked');
  setChk('resHostCache', true);
  setVal('resMode', 'tp8-dual'); fire($('resMode'), 'change');
  has(msgs(), '未实现', 'tp8-dual: blocked');
  setVal('resMode', 'auto'); fire($('resMode'), 'change');
  checkAndFire('resHostCache', false);
  has(cmd(), '--mode auto', 'auto: mode emitted');
  ok($('tp8Banner').classList.contains('hidden'), 'auto: banner hidden');
  setVal('resMode', 'ep8'); fire($('resMode'), 'change');
  checkAndFire('resHostCache', true);
  has(msgs(), '不需要 host cache', 'ep8+hc: warning');
  checkAndFire('resHostCache', false);

  // ep8-fidelity：EP P→D（device NZ，免每 token 回拷）
  w.BUILDER.PRESETS.resident_ep8f();
  w.refresh();
  ok($('resMode').value === 'ep8-fidelity', 'ep8f: mode value');
  ok($('resHostCache').checked, 'ep8f: host cache auto-on');
  c = cmd();
  has(c, '--mode ep8-fidelity', 'ep8f: mode arg');
  has(c, 'moe_host_cache.py start', 'ep8f: host cache start');
  has(c, 'decode：inc（EP P→D', 'ep8f: server decode note');
  has(msgs(), 'dispatch/combine', 'ep8f: warning');
  ok(!E().length, 'ep8f: no blocking error');
  checkAndFire('resHostCache', false);
  has(msgs(), '需要 host cache', 'ep8f no-hc: blocked');

  /* ══════════ 4) 校验 ══════════ */
  resident();
  setAndFire('cards', '0,1,2');
  has(msgs(), '需要 8 个物理卡号', 'validate: cards count');
  setAndFire('cards', '0,0,1,2,3,4,5,6');
  has(msgs(), '有重复', 'validate: duplicate cards');
  setAndFire('cards', '0,1,2,3,4,5,6,7');
  setAndFire('resWarm', '16,32,2048');
  has(msgs(), '预热阶梯最大', 'validate: ladder > pool');
  setAndFire('resWarm', 'abc');
  has(msgs(), '非法值', 'validate: bad ladder');
  setAndFire('resWarm', '16,32,64,128,256,512,1024');
  w.BUILDER.setQueue([{ prompt: '长'.repeat(1100), chat: true, thinking: 'chat', decode: 32, repeat: 1 }]);
  has(msgs(), '池容量', 'validate: prompt+decode > lmax');
  w.BUILDER.setQueue([{ prompt: '正常', chat: true, thinking: 'chat', decode: 1, repeat: 1 }]);
  setAndFire('resJob', 'a b/c');
  ok(!E().length, 'validate: job sanitized not blocking');
  has(cmd(), '--job a_b_c', 'validate: job sanitized in command');
  has(msgs(), '非法字符', 'validate: job warning');
  setAndFire('resJob', 'resident');
  checkAndFire('resSharedTmp', false);
  lacks(cmd(), 'DS41_BENCH_ROOT', 'validate: shared tmp off -> no export');
  has(msgs(), '临时目录放共享盘', 'validate: shared tmp warning');
  checkAndFire('resSharedTmp', true);
  setAndFire('resExtraEnv', 'NOPE');
  has(msgs(), '非法行', 'validate: bad env line');
  setAndFire('resExtraEnv', 'FOO=bar baz');
  has(msgs(), '非法行', 'validate: env value with space rejected');
  c = selectTab('server');
  lacks(c, 'FOO=bar baz', 'validate: unsafe env line not emitted');
  setAndFire('resExtraEnv', 'FOO=bar_baz-1.2:/x');
  ok(!E().length, 'validate: safe env line accepted');
  has(selectTab('server'), 'export FOO=bar_baz-1.2:/x', 'validate: safe env exported');
  setAndFire('resExtraEnv', '');
  selectTab('all');

  /* ══════════ 5) dry-run ══════════ */
  resident();
  checkAndFire('resDryRun', true);
  c = cmd();
  has(c, '--dry-run', 'dry-run: flag present');
  ok(count(c, '--dry-run') >= 3, 'dry-run: server + warmup + run');
  c = selectTab('teardown');
  has(c, 'export DS41_BENCH_ROOT=', 'teardown: shared tmp export');
  has(c, '# python3.11 tools/bench_cli.py status', 'dry-run: status commented');
  has(c, '# python3.11 tools/bench_cli.py stop', 'dry-run: stop commented');
  lacks(c, '\npython3.11 tools/bench_cli.py stop', 'dry-run: no live stop');
  selectTab('all');
  checkAndFire('resDryRun', false);
  w.BUILDER.PRESETS.resident_tp8(); w.refresh();
  checkAndFire('resDryRun', true);
  c = selectTab('server');
  has(c, '[dry-run] 跳过 host cache start', 'dry-run: hc skipped');
  lacks(c, '\npython3.11 tools/moe_host_cache.py start', 'dry-run: no hc start');
  checkAndFire('resDryRun', false);
  selectTab('all');

  /* ══════════ 6) 一键脚本 ══════════ */
  resident();
  checkAndFire('resBackground', true);
  ok(tabIds().includes('oneclick'), 'oneclick: tab present');
  c = selectTab('oneclick');
  has(c, 'nohup python3.11 tools/bench_server.py start', 'oneclick: nohup server');
  has(c, 'trap cleanup EXIT INT TERM', 'oneclick: trap');
  has(c, 'SOCK="$BENCH_ROOT/resident/server.sock"', 'oneclick: socket path');
  has(c, 'for i in $(seq 1', 'oneclick: wait loop');
  has(c, 'kill -0 "$SERVER_PID"', 'oneclick: alive check');
  has(c, 'bench_cli.py warmup', 'oneclick: warmup');
  has(c, 'bench_cli.py run', 'oneclick: run');
  has(c, 'bench_cli.py status', 'oneclick: status');
  checkAndFire('resBackground', false);
  ok(!tabIds().includes('oneclick'), 'oneclick: tab removed when off');
  selectTab('all');

  /* ══════════ 7) launch ══════════ */
  w.setEntry('launch'); w.BUILDER.PRESETS.smoke(); w.refresh();
  c = cmd();
  has(c, 'launch_mini.py', 'launch smoke: entry');
  has(c, '--config configs/mini_ds41_flash.yaml', 'launch smoke: config');
  has(c, '--seq 2048', 'launch smoke: seq');
  has(c, '--iters 1', 'launch smoke: iters');
  lacks(c, '--prof ', 'launch smoke: prof off');
  ok(!E().length, 'launch smoke: clean');
  ok(!d.querySelector('.perf-rl').classList.contains('hidden'),
     'launch: resident replay switches visible');
  setAndFire('batches', '2'); w.refresh();
  c = cmd();
  has(c, '--batches 2', 'launch multi-batch: batches');
  lacks(c, '--batch-mode', 'launch multi-batch: no --batch-mode (packed is the only impl)');
  setAndFire('batches', '1'); w.refresh();
  w.BUILDER.PRESETS.decode(); w.refresh();
  c = cmd();
  has(c, '--decode 8', 'launch decode: steps');
  has(c, '--decode-mode inc', 'launch decode: mode');
  has(c, 'DS41M_PD_MERGE=1', 'launch decode: pd merge');
  has(c, 'DS41M_DECODE_LAYOUT=ep', 'launch decode: layout');
  let un = (c.match(/^unset .*$/m) || [''])[0];
  ok(/DS41M_TP_DECODE_ONLY/.test(un) && /DS41M_MOE_CHUNK/.test(un) &&
     /DS41M_DECODER_REPLAY/.test(un), 'launch decode: stale managed env explicitly unset');
  ok(!/DS41M_PD_MERGE\b/.test(un), 'launch decode: exported PD_MERGE not unset');
  ok(!/DS41M_MOEMODE\b/.test(un), 'launch decode: exported MOEMODE not unset');
  setAndFire('pdMode', 'off', 'change'); w.refresh();
  has(msgs(), 'PREP_CACHE_N', 'launch: PD off + deepgemm warns transform cache');
  c = cmd();
  un = (c.match(/^unset .*$/m) || [''])[0];
  ok(/DS41M_PD_MERGE\b/.test(un) && /DS41M_TP_DECODE_ONLY\b/.test(un),
     'launch PD off: prefill-reuse really explicit (stale PD vars unset)');
  lacks(c, 'DS41M_PD_MERGE=1', 'launch PD off: no pd merge export');
  setAndFire('pdMode', 'ep', 'change'); w.refresh();
  w.BUILDER.PRESETS.prompt_decode(); w.refresh();
  c = cmd();
  has(c, '--seq 0', 'launch prompt: seq 0');
  has(c, "--prompt '请用一句话介绍杭州", 'launch prompt: prompt');
  has(c, '--chat', 'launch prompt: chat template by default');
  setAndFire('promptMode', 'thinking', 'change'); w.refresh();
  has(cmd(), '--thinking thinking', 'launch prompt: thinking mode');
  setAndFire('promptMode', 'raw', 'change'); w.refresh();
  lacks(cmd(), '--chat', 'launch prompt: raw mode omits --chat');
  setAndFire('promptMode', 'chat', 'change'); w.refresh();
  w.setEntry('launch'); w.BUILDER.PRESETS.smoke();
  setAndFire('weightsPath', '/tmp/only-launch-w');
  setAndFire('extraArgs', '--allow-placeholder');
  has(cmd(), '--weights-path /tmp/only-launch-w', 'scope: launch weights path');
  w.setEntry('real'); w.BUILDER.PRESETS.real(); w.refresh();
  lacks(cmd(), '/tmp/only-launch-w', 'scope: real ignores launch weights');
  w.setEntry('resident'); w.BUILDER.PRESETS.resident(); w.refresh();
  lacks(cmd(), '--allow-placeholder', 'scope: resident ignores launch extra args');
  lacks(cmd(), '/tmp/only-launch-w', 'scope: resident ignores launch weights');

  /* ══════════ 7b) 性能跳过开关（跳 MoE / 跳 engram）══════════ */
  w.setEntry('launch'); w.BUILDER.PRESETS.smoke(); w.refresh();
  lacks(cmd(), 'export DS41M_SKIP_MOE', 'skip: off by default');
  checkAndFire('skipMoe', true); checkAndFire('skipEngram', true);
  c = cmd();
  has(c, 'DS41M_SKIP_MOE=1', 'skip: emits skip-MoE env');
  has(c, 'DS41M_SKIP_ENGRAM=1', 'skip: emits skip-engram env');
  w.setEntry('hit'); w.BUILDER.PRESETS.perf(); w.refresh();
  checkAndFire('skipMoe', true); w.refresh();
  has(cmd(), 'DS41M_SKIP_MOE=1', 'skip: also in hit script');
  w.setEntry('real'); w.BUILDER.PRESETS.real(); w.refresh();
  checkAndFire('skipMoe', true); w.refresh();
  has(cmd(), 'DS41M_SKIP_MOE=1', 'skip: also in real script');
  has(msgs(), '只用于性能', 'skip: real warns perf-only');
  checkAndFire('skipMoe', false); w.refresh();
  lacks(cmd(), 'export DS41M_SKIP_MOE', 'skip: toggling off removes env');

  /* ══════════ 7c) 设置面板（CANN / TMPDIR / 路径 / 真权重）══════════ */
  ok(!!$('settingsBtn') && !!$('settingsModal'), 'settings: entry + modal');
  setAndFire('setCann', '/opt/cann-9.2');
  setAndFire('setRuntime', '/opt/ds41-rt');
  setAndFire('setTmpdir', '/mnt/xds/tmp');
  setAndFire('setOutRoot', '/mnt/xds/out');
  setAndFire('setTraceRoot', '/mnt/xds/prof');
  setAndFire('setRealWeights', '/mnt/xds/real-w');
  setAndFire('setWorkdir', '/mnt/xds/ds41_mini');
  $('setApply').dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok($('outRoot').value === '/mnt/xds/out', 'settings: out root applied');
  ok($('launchProfDir').value === '/mnt/xds/prof/launch', 'settings: launch trace root');
  ok($('hitProfDir').value === '/mnt/xds/prof/{L}', 'settings: hit trace root');
  ok($('pointsProfDir').value === '/mnt/xds/prof/{L}', 'settings: points trace root');
  w.setEntry('launch'); w.refresh();
  c = cmd();
  has(c, 'cd /mnt/xds/ds41_mini', 'settings: workdir in script');
  has(c, 'export DS41_CANN=/opt/cann-9.2', 'settings: CANN exported before source');
  has(c, 'export DS41_PACK_RUNTIME=/opt/ds41-rt', 'settings: runtime exported');
  has(c, 'export TMPDIR=/mnt/xds/tmp', 'settings: TMPDIR exported');
  has(w.BUILDER.getState().settings.setCann, '/opt/cann-9.2', 'settings: exported in state');
  w.setEntry('real'); w.BUILDER.PRESETS.real(); w.refresh();
  has(cmd(), '--weights-path /mnt/xds/real-w', 'settings: real weights path');
  resetAll();

  // 单卡性能预设：world=1、cards=0、两个跳过都开
  w.setEntry('launch'); w.refresh();
  has($('presets').textContent, '单卡 · 跳 MoE+engram', 'skip: single-card preset visible');
  w.BUILDER.PRESETS.single_skip(); w.refresh();
  ok($('world').value === '1', 'skip: preset world=1');
  ok($('cards').value === '0', 'skip: preset cards=0');
  ok($('skipMoe').checked && $('skipEngram').checked, 'skip: preset turns both on');
  c = cmd();
  has(c, '--world 1', 'skip: preset command world 1');
  has(c, '--cards 0', 'skip: preset command cards 0');
  has(c, 'DS41M_SKIP_MOE=1', 'skip: preset command skips MoE');
  has(c, 'DS41M_SKIP_ENGRAM=1', 'skip: preset command skips engram');
  has(c, '--seq 2048', 'skip: preset seq');
  has(c, '--iters 2', 'skip: preset iters');

  // 从多点页面的新手引导一键切到单卡预设
  resetAll(); w.setEntry('points'); w.refresh();
  $('qsSingleCard').dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok(d.body.dataset.scenario === 'launch', 'skip: quick-start switches to launch');
  ok($('world').value === '1' && $('cards').value === '0', 'skip: quick-start world/cards');
  ok($('skipMoe').checked && $('skipEngram').checked, 'skip: quick-start both switches');
  // 多点也支持单卡：1 个卡号 + 两个跳过
  resetAll(); w.setEntry('points'); w.refresh();
  setAndFire('cards', '0'); w.refresh();
  has(msgs(), '跑不了厂商 engram', 'skip: points single card needs skip-engram');
  checkAndFire('skipMoe', true); checkAndFire('skipEngram', true); w.refresh();
  ok(!E().length, 'skip: points single card clean with both switches');
  c = cmd();
  has(c, '--cards 0', 'skip: points command cards 0');
  has(c, 'DS41M_SKIP_MOE=1', 'skip: points command skips MoE');
  has(c, 'DS41M_SKIP_ENGRAM=1', 'skip: points command skips engram');
  // 点位预设按钮：保持点位表不变，只改卡号 + 两个开关
  resetAll(); w.setEntry('points'); w.refresh();
  const _ptRows0 = d.querySelectorAll('#pointsTbody tr').length;
  d.querySelector('#pointsPresets [data-points-preset="singlecard"]')
    .dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok($('cards').value === '0' && $('skipMoe').checked && $('skipEngram').checked,
     'skip: points single-card preset');
  ok(d.querySelectorAll('#pointsTbody tr').length === _ptRows0,
     'skip: points single-card preset keeps rows');
  resetAll();

  /* ══════════ 8) hit ══════════ */
  w.setEntry('hit'); w.BUILDER.PRESETS.perf(); w.refresh();
  c = cmd();
  has(c, 'run_hit.sh', 'hit: entry');
  has(c, '-L 16k', 'hit: length');
  has(c, '--hit 0.97', 'hit: ratio');
  has(c, '--iters 4', 'hit: iters');
  has(c, '--prof-ranks 0', 'hit: prof ranks');
  has(c, '-o /mnt/xds/sfs/l00972766/tmp/bench_hit16k', 'hit: out');
  ok(!E().length, 'hit: clean');
  ok(d.querySelector('.perf-rl').classList.contains('hidden'),
     'hit: resident replay switches hidden');
  checkAndFire('plain', true);
  c = cmd();
  has(c, '--plain', 'hit plain: flag');
  lacks(c, '--hit 0.97', 'hit plain: no hit ratio');
  checkAndFire('plain', false);
  setAndFire('hitRatio', 'abc');
  has(msgs(), '命中率写法不认', 'hit: invalid ratio');
  setAndFire('hitRatio', '97%');
  ok(!E().length, 'hit: 97% accepted');
  checkAndFire('hitDryRun', true);
  has(cmd(), '--dry-run', 'hit: dry-run');
  checkAndFire('hitDryRun', false);
  checkAndFire('hitNoBar', true);
  has(cmd(), '--no-bar', 'hit: no-bar');
  checkAndFire('hitNoBar', false);
  setAndFire('cards', '0,1,2,3,4,5,6,7,8');
  has(msgs(), '支持 1..8 个卡号', 'hit: cards count (9 > 8)');
  setAndFire('cards', '0');
  has(msgs(), '跑不了厂商 engram', 'hit: single card needs skip-engram');
  checkAndFire('skipEngram', true);
  ok(!E().length, 'hit: single card + skip engram ok');
  setAndFire('cards', '0,1,2,3,4,5,6,7');
  checkAndFire('skipEngram', false);

  /* ══════════ 9) real ══════════ */
  w.setEntry('real'); w.BUILDER.PRESETS.real(); w.refresh();
  c = cmd();
  has(c, 'run_real.sh', 'real: entry');
  lacks(c, 'tools/selfcheck.py', 'selfcheck: real script does not embed it');
  has(c, '-L 16k', 'real: length');
  has(c, '--mode prefill', 'real: mode');
  has(c, '--ids-file /mnt/xds/sfs/l00972766/parity/ids_16k.pt', 'real: ids');
  has(c, '--ref /mnt/xds/sfs/l00972766/parity/ref', 'real: ref');
  has(c, '--alop-config configs/alop_mindstudio.yaml', 'real: alop');
  has(msgs(), '145 ms', 'real moemode: deepgemm perf warning');
  has(c, '-o /mnt/xds/sfs/l00972766/tmp/bench_real16k', 'real: out');
  ok(!E().length, 'real: clean');
  setAndFire('realMode', 'both', 'change');
  setAndFire('realLayers', '15');
  has(msgs(), '互斥', 'real: layers vs hit/both');
  setAndFire('realLayers', '0');
  setAndFire('realHitRatio', '97');
  has(msgs(), '只认 0–1', 'real: ratio must be fraction');
  setAndFire('realHitRatio', '0.96875');
  checkAndFire('realNoPrecision', true);
  has(cmd(), '--no-precision', 'real: no-precision');
  checkAndFire('realNoPrecision', false);
  checkAndFire('realParity', true);
  c = cmd();
  has(c, '--parity', 'real: parity');
  has(c, '--collect-ranks 0', 'real: collect ranks');
  checkAndFire('realParity', false);
  checkAndFire('realDryRun', true);
  has(cmd(), '--dry-run', 'real: dry-run');
  checkAndFire('realDryRun', false);
  checkAndFire('realProfOnly', true);
  checkAndFire('realNoprofOnly', true);
  has(msgs(), '不能同时勾选', 'real: contradictory prof flags');
  checkAndFire('realProfOnly', false);
  checkAndFire('realNoprofOnly', false);

  // ── 真权重卡全字段 → 命令参数（复测）──
  w.BUILDER.PRESETS.real(); w.refresh();
  c = cmd();
  ok($('realParity').checked, 'real full: parity default on');
  lacks(c, '--no-precision', 'real full: precision gate on by default');
  setAndFire('realMode', 'hit', 'change');
  setAndFire('realHitRatio', '0.96875');
  has(cmd(), '--hit 0.96875', 'real full: hit ratio');
  setAndFire('realIters', '5');
  has(cmd(), '--iters 5', 'real full: iters');
  setAndFire('realWarmup', '2');
  has(cmd(), '--warmup-iters 2', 'real full: warmup');
  setAndFire('realLayers', '3');
  has(msgs(), '互斥', 'real full: layers vs hit');
  setAndFire('realLayers', '0');
  setAndFire('realIdsFile', '/x/ids.pt');
  has(cmd(), '--ids-file /x/ids.pt', 'real full: ids');
  setAndFire('realRef', '/x/ref');
  has(cmd(), '--ref /x/ref', 'real full: ref');
  setAndFire('realWeightsPath', '/w/real');
  has(cmd(), '--weights-path /w/real', 'real full: weights path');
  setAndFire('timeout', '7200');
  has(cmd(), '--timeout 7200', 'real full: timeout');
  setAndFire('console', 'rank0', 'change');
  has(cmd(), '--console rank0', 'real full: console');
  setAndFire('realCollectRanks', 'all', 'change');
  has(cmd(), '--collect-ranks all', 'real full: collect ranks');
  setAndFire('realResident', 'false', 'change');
  has(cmd(), '--lazy', 'real full: lazy');
  setAndFire('realResident', 'true', 'change');
  setAndFire('realHostCache', 'true', 'change');
  has(cmd(), '--host-cache true', 'real full: host cache');
  setAndFire('realHostCache', 'false', 'change');
  checkAndFire('realEngramHost', true);
  has(cmd(), '--engram-host', 'real full: engram host');
  checkAndFire('realEngramHost', false);
  checkAndFire('realNoMoeChunk', true);
  has(cmd(), '--no-moe-chunk', 'real full: no moe chunk');
  checkAndFire('realNoMoeChunk', false);
  checkAndFire('realNoAcCache', true);
  has(cmd(), '--no-ac-cache', 'real full: no ac cache');
  checkAndFire('realNoAcCache', false);
  checkAndFire('realForce', true);
  has(cmd(), '--force', 'real full: force');
  checkAndFire('realForce', false);
  setAndFire('realProfDir', '/tmp/prof');
  has(cmd(), '-P /tmp/prof', 'real full: prof dir');
  setAndFire('realProfDir', '');
  checkAndFire('realNoBar', true);
  has(cmd(), '--no-bar', 'real full: no bar');
  checkAndFire('realNoBar', false);
  setAndFire('realBarEvery', '15'); setAndFire('realBarFd', '7');
  has(cmd(), '--bar-every 15', 'real full: bar every');
  has(cmd(), '--bar-fd 7', 'real full: bar fd');
  setAndFire('realBarEvery', '30'); setAndFire('realBarFd', '9');
  setAndFire('realAlopConfig', 'builtin', 'change');
  has(cmd(), '--no-alop-config', 'real full: builtin alop');
  setAndFire('realAlopConfig', 'mindstudio', 'change');
  setAndFire('realExtraArgs', '--foo 1');
  has(cmd(), '--foo 1', 'real full: extra args');
  setAndFire('realExtraArgs', '');
  checkAndFire('realProf', false);
  has(cmd(), '--noprof-only', 'real full: prof off → noprof-only');
  checkAndFire('realProf', true);
  checkAndFire('realProfOnly', true);
  has(cmd(), '--prof-only', 'real full: prof-only');
  checkAndFire('realProfOnly', false);
  checkAndFire('realNoprofOnly', true);
  has(cmd(), '--noprof-only', 'real full: noprof-only');
  checkAndFire('realNoprofOnly', false);
  w.BUILDER.PRESETS.real(); w.refresh();

  /* ══════════ 10) 高级 ══════════ */
  resident();
  w.BUILDER.setUIMode('advanced');
  ok(d.body.dataset.uiMode === 'advanced', 'advanced: body dataset');
  setAndFire('envDS41M_SERVE_DECODE', 'full', 'change');
  setAndFire('envDS41M_MOE_CHUNK', '1024', 'change');
  c = selectTab('server');
  has(c, 'export DS41M_SERVE_DECODE=full', 'advanced: serve decode env');
  has(c, 'export DS41M_MOE_CHUNK=1024', 'advanced: moe chunk env');
  setAndFire('envDS41M_DECODE_GRAPH', '1', 'change');
  setAndFire('envDS41M_DECODE_GRAPH_CUT', '15', 'input');
  c = selectTab('server');
  has(c, 'export DS41M_DECODE_GRAPH=1', 'advanced: decode graph env');
  has(c, 'export DS41M_DECODE_GRAPH_CUT=15', 'advanced: decode graph cut');
  has(msgs(), 'decode graph', 'advanced: decode graph note');
  setAndFire('envDS41M_DECODE_GRAPH', '', 'change');
  setAndFire('envDS41M_DECODE_GRAPH_CUT', '', 'input');

  // DSpark（投机解码）：开关 / 导出 / tp8 约束 / verify 图联动
  ok(!!$('envDS41M_DSPARK'), 'advanced: DSpark field exists');
  setAndFire('envDS41M_DSPARK', '5', 'change');
  ok(!msgs().includes('需要 tp8'), 'advanced: DSpark works on ep8 (not tp8-only)');
  has(msgs(), '不用 P→D 切权重', 'advanced: ep8 DSpark explains no weight switch');
  setAndFire('envDS41M_SERVE_DECODE', 'full', 'change');
  has(msgs(), '需要增量 decode', 'advanced: DSpark rejects full decode');
  setAndFire('envDS41M_SERVE_DECODE', '', 'change');
  setAndFire('resMode', 'tp8-fidelity', 'change');
  c = selectTab('server');
  has(c, 'export DS41M_DSPARK=5', 'advanced: DSpark export');
  has(msgs(), 'DS41M_DECODE_GRAPH=1', 'advanced: DSpark verify graph needs decode graph');
  setAndFire('envDS41M_DSPARK_VERIFY_GRAPH', '0', 'change');
  has(msgs(), 'verify 图已手动关闭', 'advanced: DSpark verify graph off warning');
  setAndFire('envDS41M_DSPARK_VERIFY_GRAPH', '', 'change');
  setAndFire('envDS41M_DSPARK', '', 'change');
  setAndFire('resMode', 'ep8', 'change');
  ok(!d.getElementById('perLayerCache'), 'advanced: internal env not exposed');
  lacks(c, 'DS41M_PERLAYER_CACHE', 'advanced: no internal env in script');
  setAndFire('envDS41M_SERVE_DECODE', 'inc', 'change');
  has(msgs(), 'NaN 根因已修', 'advanced: inc warning');
  setAndFire('envDS41M_SERVE_DECODE', '', 'change');
  checkAndFire('replay', false);
  has(msgs(), 'DS41_ALLOW_SUBOPTIMAL', 'advanced: degradation needs reason');
  setAndFire('suboptimalReason', 'ab-replay-off');
  c = selectTab('server');
  has(c, "export DS41_ALLOW_SUBOPTIMAL='ab-replay-off'", 'advanced: suboptimal export');
  checkAndFire('replay', true);
  setAndFire('suboptimalReason', '');
  checkAndFire('resClientVerbose', true);
  checkAndFire('resClientNoColor', true);
  c = selectTab('client');
  has(c, '--verbose', 'advanced: client verbose');
  has(c, '--no-color', 'advanced: client no-color');
  checkAndFire('resClientVerbose', false);
  checkAndFire('resClientNoColor', false);
  setAndFire('resClientRunMode', 'ep8', 'change');
  has(selectTab('client'), '--mode ep8', 'advanced: client run mode');
  setAndFire('resClientRunMode', '', 'change');
  checkAndFire('resCmdHostDetach', true);
  has(selectTab('teardown'), 'hostcache-detach', 'advanced: hostcache-detach');
  checkAndFire('resCmdHostDetach', false);
  w.BUILDER.setUIMode('simple');
  selectTab('all');

  /* ══════════ 11) 导入 / 导出 ══════════ */
  resident();
  w.BUILDER.setQueue([
    { prompt: '导入第一', chat: true, thinking: 'chat', decode: 12, repeat: 2 },
    { prompt: '导入第二', chat: false, thinking: 'thinking', decode: 6, repeat: 1 },
  ]);
  setAndFire('resJob', 'impjob');
  setAndFire('resLMax', '2048');
  const st = w.BUILDER.getState();
  ok(st.v === 1, 'state: version 1');
  ok(st.resident.resJob === 'impjob', 'state: resident job');
  ok(st.queue.length === 2 && st.queue[0].prompt === '导入第一', 'state: queue exported');
  resetAll();
  ok($('resJob').value === 'resident', 'state: reset before import');
  w.BUILDER.setState(st);
  w.refresh();
  ok($('resJob').value === 'impjob' && $('resLMax').value === '2048', 'state: scalar round-trip');
  ok(queueRows().length === 2 && d.querySelectorAll('#resQueue .q-prompt')[1].value === '导入第二',
     'state: queue round-trip');
  w.BUILDER.setState({ v: 1, resident: { resMode: 'bogus', resLMax: 'not-a-number' } });
  w.refresh();
  ok($('resMode').value !== 'bogus', 'state: unknown select value ignored');
  const url = w.BUILDER.shareUrl();
  has(url, '#cfg=', 'share: url has cfg');
  w.BUILDER.applyAllDefaults();
  w.location.hash = url.split('#')[1];
  ok(w.BUILDER.loadFromHash(), 'share: loadFromHash ok');
  w.refresh();
  ok($('resJob').value === 'impjob', 'share: state restored from hash');
  ok(queueRows().length === 2, 'share: queue restored from hash');
  w.BUILDER.importJsonText('{bad json', 'test');
  has($('ioStatus').textContent, '导入失败', 'import: bad JSON handled');
  w.BUILDER.importJsonText(JSON.stringify({ v: 99, resident: { resJob: 'v99' } }), 'test');
  has($('ioStatus').textContent, '导入完成', 'import: wrong version tolerated');
  has($('ioStatus').textContent, '不是 1', 'import: version note');
  w.BUILDER.importJsonText(JSON.stringify({ v: 1, queue: [] }), 'test');
  has(msgs(), '请求队列为空', 'import: empty queue blocks after import');

  /* ══════════ 12) 预设重置 / 高亮 ══════════ */
  w.setEntry('launch');
  w.BUILDER.PRESETS.smoke();
  w.refresh();
  setAndFire('seq', '9999');
  w.BUILDER.PRESETS.smoke();
  w.refresh();
  ok($('seq').value === '2048', 'preset: resets previous value');
  setAndFire('weightsPath', '/tmp/w-leak');
  w.BUILDER.PRESETS.prompt_decode();
  w.refresh();
  ok($('weightsPath').value === '', 'preset: no old value leak');
  lacks(cmd(), '/tmp/w-leak', 'preset: leak not in command');
  const smokeBtn = d.querySelector('#presets button[data-preset="smoke"]');
  smokeBtn.dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(d.querySelector('#presets button[data-preset="smoke"]').classList.contains('on'),
     'preset: highlight after click');
  setAndFire('seq', '4096');
  ok(!d.querySelector('#presets button.on'), 'preset: highlight cleared on edit');
  for (const name of Object.keys(w.PRESETS)) {
    try { w.PRESETS[name](); w.refresh(); pass++; }
    catch (e) { fail++; errors.push('FAIL: preset ' + name + ': ' + e.message); }
  }
  resident();

  /* ══════════ 13) 多点：自定义 / 导入 / replay_tail / moemode / 两条命令 ══════════ */
  function ptRows() { return [...d.querySelectorAll('#pointsTbody tr')]; }
  function ptSet(i, cls, v) {
    const el = ptRows()[i].querySelector(cls);
    el.value = v;
    fire(el, 'input');
  }
  function clickId(id) { $(id).dispatchEvent(new w.Event('click', { bubbles: true })); }
  function flatScript(text) {
    // 续行合并 + 压掉续行缩进产生的多余空格
    return text.replace(/\\\n/g, ' ')
      .split('\n').map(l => l.replace(/[ \t]+/g, ' ').trim()).join('\n');
  }
  function pointRuns(text) {
    return flatScript(text).split('\n').filter(l => l.includes('bash tools/run_hit.sh --points'));
  }

  w.setEntry('points');
  d.querySelector('#pointsPresets button[data-points-preset="custom"]')
    .dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok(ptRows().length === 2, 'points custom: starts with 2 blank rows');
  ok(ptRows()[0].querySelector('.pt-l').value === '', 'points custom: L blank');
  ok(ptRows()[0].querySelector('.pt-h').value === '', 'points custom: hit blank');
  ok(ptRows()[0].querySelector('.pt-note'), 'points table: note column exists');

  // v2：一行 = 一个批次；点位名 = 分组
  ok(ptRows()[0].querySelector('.pt-g'), 'points table: group column exists');
  ptSet(0, '.pt-g', 'p1'); ptSet(0, '.pt-l', '81920'); ptSet(0, '.pt-h', '0.96');
  ptSet(0, '.pt-note', '备注A');
  ptSet(1, '.pt-g', 'p1'); ptSet(1, '.pt-l', '16384'); ptSet(1, '.pt-h', '97%');

  // 自定义行：增 / 上移 / 删
  clickId('pointsAdd');
  ok(ptRows().length === 3, 'points custom: add row');
  ptSet(2, '.pt-l', '32768'); ptSet(2, '.pt-h', '0.96');
  ptRows()[2].querySelector('[data-pt-op="up"]')
    .dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok(ptRows().length === 3 && ptRows()[1].querySelector('.pt-l').value === '32768',
     'points custom: move row up');
  ptRows()[1].querySelector('[data-pt-op="del"]')
    .dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  ok(ptRows().length === 2, 'points custom: delete row');
  ok(ptRows()[0].querySelector('.pt-note').value === '备注A',
     'points custom: note survives row ops');

  // JSON v2：同点位名的两行合成一个 point，batches 里两条 {L,hit}
  let pv = $('pointsJsonPreview').textContent;
  ok(Array.isArray(JSON.parse(pv)), 'points json: preview is valid JSON');
  has(pv, '"tag": "p1"', 'points json: point tag exported');
  has(pv, '"batches"', 'points json: batches array');
  has(pv, '"L": 81920', 'points json: first batch L');
  has(pv, '"L": 16384', 'points json: second batch L');
  lacks(pv, '备注A', 'points json: note is UI-only');
  has(pv, '"replay": true', 'points json: per-point replay exported (default on)');
  ok(!$('pointsCopyJson').disabled && !$('pointsCopyNoprof').disabled,
     'points: valid table enables copy');

  // 行内错误高亮 + 阻断复制（L 非 128 倍数）
  ptSet(0, '.pt-l', '65535');
  ok(ptRows()[0].querySelector('.pt-l').classList.contains('bad'),
     'points: L%128 error highlighted');
  ok($('pointsCopyJson').disabled && $('pointsCopyNoprof').disabled && $('pointsCopyProf').disabled,
     'points: invalid L blocks copy');
  ok($('msgsCard') && !$('msgsCard').open, 'msgs: fold panel closed by default');
  has($('msgsBadge').textContent, '错误', 'msgs: badge shows error count');
  has($('pointsStatus').textContent, '第 1 行 · L', 'points: error names row + field');
  ptSet(0, '.pt-l', '81920');
  ok(!$('pointsCopyJson').disabled, 'points: fixing row re-enables copy');
  ok(!$('msgsBadge').textContent.includes('错误'), 'msgs: badge clears after fix');

  // 粘贴导入：v2 格式 + 未知键提示
  setVal('pointsImportText',
    '[{"tag":"a","batches":[{"L":81920,"hit":0.96},{"L":81920,"hit":"90%"}],"foo":1},' +
    ' {"L":16384,"hit":0.97}]');
  clickId('pointsImportPaste');
  ok(ptRows().length === 3, 'points import v2: batches expanded + legacy point kept');
  has($('pointsImportStatus').textContent, '忽略未知键 foo', 'points import: unknown key noted');
  ok($('pointsImportStatus').classList.contains('warn'), 'points import: warning status');
  ok($('pointsPath').value === './points_custom.json', 'points import paste: custom path');
  pv = $('pointsJsonPreview').textContent;
  lacks(pv, 'foo', 'points import: unknown key not exported');
  has(pv, '"tag": "a"', 'points import v2: tag kept');
  has(pv, '"hit": 0.9', 'points import v2: percent normalized');

  // 粘贴导入：严格校验定位到行 + 项
  setVal('pointsImportText',
    '[{"batches":[{"L":1000,"hit":0.96}]},{"batches":[{"L":81920,"hit":150}]}]');
  clickId('pointsImportPaste');
  const ist = $('pointsImportStatus').textContent;
  has(ist, '第 1 个点位 第 1 批 · L', 'points import: bad L located');
  has(ist, '第 2 个点位 第 1 批 · hit', 'points import: bad hit located');
  ok($('pointsImportStatus').classList.contains('err'), 'points import: error status');
  setVal('pointsImportText', '{bad json');
  clickId('pointsImportPaste');
  has($('pointsImportStatus').textContent, 'JSON 解析失败', 'points import: bad JSON handled');

  // 上传导入：旧格式 batches:2 → 展开成 2 行（同一分组）
  const OldFR = w.FileReader;
  w.FileReader = function () {
    this.readAsText = f => {
      this.result = f.__text;
      if (this.onload) this.onload();
    };
  };
  const fileInput = $('pointsImportFile');
  Object.defineProperty(fileInput, 'files', {
    configurable: true,
    value: [{ name: 'my_points.json',
      __text: '[{"L":8192,"hit":0.96,"batches":2,"replay":true,"replay_tail":8}]' }],
  });
  fileInput.dispatchEvent(new w.Event('change', { bubbles: true }));
  w.FileReader = OldFR;
  ok(ptRows().length === 2, 'points import upload: legacy batches:N expanded');
  ok($('pointsPath').value === './my_points.json', 'points import upload: path from filename');
  has($('pointsImportStatus').textContent, '文件', 'points import upload: status');
  ok(count($('pointsJsonPreview').textContent, '"L": 8192') === 2,
     'points import upload: two batches in one point');

  // 分享 URL 往返：点位名 / 备注都要保留
  ptSet(0, '.pt-note', '分享备注');
  const pst = w.BUILDER.getState();
  ok(pst.points.rows[0].note === '分享备注', 'points state: note exported');
  ok(typeof pst.points.rows[0].group === 'string', 'points state: group exported');
  const purl = w.BUILDER.shareUrl();
  w.BUILDER.applyAllDefaults();
  w.location.hash = purl.split('#')[1];
  ok(w.BUILDER.loadFromHash(), 'points share: loadFromHash');
  w.refresh();
  const prst = w.BUILDER.getState().points.rows;
  ok(prst[0].note === '分享备注', 'points share: note round-trip');
  ok(ptRows().length === 2, 'points share: table restored');

  // ── per-DP 编辑器（与多点位互斥，2026-10-21 已开放）──
  const dpRows = () => [...d.querySelectorAll('#perdpTbody tr')];
  function dpSet(i, v) {
    const el = dpRows()[i].querySelector('.dp-spec');
    el.value = v; fire(el, 'input');
  }
  ok(!$('pointsModeBox').classList.contains('hidden'), 'per-DP: 模式入口可见');
  fire($('pointsModePerdp'), 'change');
  ok($('pointsMultiWrap').classList.contains('hidden'), 'per-DP: multi editor hidden');
  ok(!$('perdpWrap').classList.contains('hidden'), 'per-DP: editor shown');
  ok(dpRows().length === 8, 'per-DP: 8 rank rows');
  has(dpRows()[0].querySelector('.dp-spec').title, '逗号', 'per-DP: spec format tooltip');
  has($('pointsStatus').textContent, 'DP0：没有规格', 'per-DP: empty rows blocked');
  ok([...$('pointsDelivery').options].find(o => o.value === 'spec').disabled,
     'per-DP: --points-spec disabled');
  ok($('pointsDelivery').value !== 'spec', 'per-DP: delivery falls back from spec');
  dpSet(0, '8192@0.96');
  for (let i = 1; i < 8; i++) dpSet(i, i === 1 ? '16384@97%' : '8192@0.96');
  const pdv = $('pointsJsonPreview').textContent;
  has(pdv, '"dps"', 'per-DP: json has dps');
  has(pdv, '"rank": 0', 'per-DP: json has rank');
  has(pdv, '"L": 16384', 'per-DP: per-rank L parsed');
  has(pdv, '"hit": 0.97', 'per-DP: percent hit normalized');
  ok(!/✗/.test($('pointsStatus').textContent), 'per-DP: valid after fill');
  setAndFire('pointsDelivery', 'json', 'change');
  const pdc = selectTab('cmd');
  has(pdc, '--points-json', 'per-DP: command uses inline json');
  has(pdc, '"dps"', 'per-DP: command carries dps');
  has(pdc, 'per-DP 模式', 'per-DP: script header notes mode');
  has(pdv, '"replay": true', 'per-DP: replay exported (default on)');
  setChk('perdpReplay', false); fire($('perdpReplay'), 'change');
  has($('pointsJsonPreview').textContent, '"replay": false', 'per-DP: replay can be turned off');
  setChk('perdpReplay', true); fire($('perdpReplay'), 'change');
  clickId('perdpFillAll');
  ok(dpRows()[7].querySelector('.dp-spec').value === '8192@0.96', 'per-DP: fill all rows');
  has(msgs(), '所有 DP 的规格完全一致', 'per-DP: identical specs warned');
  dpSet(2, '8192@150');
  has($('pointsStatus').textContent, 'DP2', 'per-DP: bad hit names rank');
  ok(dpRows()[2].querySelector('.dp-spec').classList.contains('bad'),
     'per-DP: bad input highlighted');
  fire($('pointsModeMulti'), 'change');
  ok(!$('pointsMultiWrap').classList.contains('hidden'), 'per-DP: multi editor back');
  const dpsJson = JSON.stringify([{ tag: 'perdp', dps: [
    { rank: 0, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 1, batches: [{ L: 16384, hit: 0.97 }] },
    { rank: 2, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 3, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 4, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 5, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 6, batches: [{ L: 8192, hit: 0.96 }] },
    { rank: 7, batches: [{ L: 8192, hit: 0.96 }] },
  ] }]);
  setVal('pointsImportText', dpsJson);
  clickId('pointsImportPaste');
  ok(!$('perdpWrap').classList.contains('hidden'), 'per-DP import: switches mode');
  ok(dpRows()[1].querySelector('.dp-spec').value === '16384@0.97',
     'per-DP import: rank1 spec filled');
  fire($('pointsModeMulti'), 'change');
  ok(ptRows().length === 2, 'per-DP: multi table preserved');
  setAndFire('pointsDelivery', 'file', 'change');

  // 完整脚本：①② 都是可执行行，参数口径正确
  setAndFire('pointsProfDir', '/mnt/xfs/prof_{L}');
  const pscript = selectTab('cmd');
  const runs = pointRuns(pscript);
  ok(runs.length === 2, 'points script: two run commands');
  ok(runs.every(l => !l.trim().startsWith('#')), 'points script: both uncommented');
  has(pscript, '两条都会顺序执行', 'points script: top explanation');
  has(pscript, '# ── ①', 'points script: separator 1');
  has(pscript, '# ── ②', 'points script: separator 2');
  has(runs[0], '--noprof-only', 'points script: ① noprof-only');
  lacks(runs[0], '--prof-only', 'points script: ① not prof-only');
  has(runs[1], '--prof-only', 'points script: ② prof-only');
  has(runs[1], '--prof-warmup 2', 'points script: ② prof-warmup');
  has(runs[1], '--prof-ranks all', 'points script: ② prof-ranks');
  ok(runs.every(l => l.includes('-o ')), 'points script: both have -o');
  has(runs[1], "-P '/mnt/xfs/prof_{L}'", 'points script: ② carries -P');
  lacks(runs[0], ' -P ', 'points script: ① no -P');
  setAndFire('pointsProfDir', '');
  setAndFire('pointsReplayTail', '256');
  const runsRt = pointRuns(selectTab('cmd'));
  has(runsRt[0], '--replay-tail 256', 'points script: global replay-tail');
  setAndFire('pointsReplayTail', '140');

  // 点位级 replay：每行可选开/关；全部关才给全局 --no-replay
  function setPtReplay(v) {
    ptRows().forEach(tr => {
      const cb = tr.querySelector('.pt-r');
      if (cb) { cb.checked = (v === '1'); fire(cb, 'change'); }
    });
  }
  ok(ptRows().every(tr => tr.querySelector('.pt-r')), 'points replay: per-row control exists');
  setPtReplay('0');
  has(pointRuns(selectTab('cmd'))[0], '--no-replay', 'points replay: all off → --no-replay');
  setPtReplay('1');
  lacks(pointRuns(selectTab('cmd'))[0], '--no-replay', 'points replay: any on → no global flag');
  // 同一点位（同组）内 replay 不一致 → 报错
  ptSet(0, '.pt-g', 'g1'); ptSet(1, '.pt-g', 'g1');
  setPtReplay('1');
  const _r0 = ptRows()[0].querySelector('.pt-r'); _r0.checked = false; fire(_r0, 'change');
  has($('pointsStatus').textContent, 'Replay 与同组其他行不一致', 'points replay: mixed group blocked');
  _r0.checked = true; fire(_r0, 'change');
  setAndFire('pointsLogLevel', 'pure', 'change');
  has(pointRuns(selectTab('cmd'))[0], '--log-level pure', 'points collection: log level pure');
  setAndFire('pointsLogLevel', 'info', 'change');
  lacks(pointRuns(selectTab('cmd'))[0], '--log-level', 'points collection: info is default (omitted)');

  // 运行轮次：三选一；完整脚本只生成所选轮次，预览同步置灰
  ok($('pointsRounds'), 'points rounds: selector exists');
  ok($('pointsRounds').value === 'both', 'points rounds: default both');
  setAndFire('pointsProfDir', '/mnt/xfs/prof_{L}');
  setAndFire('pointsWarmup', '5');
  setAndFire('pointsProfRanks', '0,3');
  setAndFire('pointsRounds', 'noprof', 'change');
  const rNp = pointRuns(selectTab('cmd'));
  ok(rNp.length === 1 && rNp[0].includes('--noprof-only'),
     'points rounds noprof: only steady command');
  lacks(selectTab('cmd'), '--prof-only', 'points rounds noprof: no trace command');
  has(selectTab('cmd'), '只执行 ①', 'points rounds noprof: explanation');
  ok($('pointsCmdBoxProf').classList.contains('is-off') &&
     !$('pointsCmdBoxNoprof').classList.contains('is-off'),
     'points rounds noprof: prof preview greyed');
  ok($('pointsCopyProf').disabled && !$('pointsCopyNoprof').disabled,
     'points rounds noprof: only steady copy enabled');
  lacks(rNp[0], ' -P ', 'points rounds noprof: no -P');
  w.__copied = '';
  clickId('pointsCopyNoprof');
  ok(String(w.__copied).includes('--noprof-only') &&
     !String(w.__copied).includes('--prof-only'),
     'points rounds noprof: steady preview copies');
  w.__copied = '';
  clickId('pointsCopyProf');
  ok(w.__copied === '', 'points rounds noprof: disabled preview cannot copy');

  setAndFire('pointsRounds', 'prof', 'change');
  const scriptProf = selectTab('cmd');
  const rPr = pointRuns(scriptProf);
  ok(rPr.length === 1 && rPr[0].includes('--prof-only'),
     'points rounds prof: only trace command');
  lacks(scriptProf, '--noprof-only', 'points rounds prof: no steady command');
  has(scriptProf, '只执行 ②', 'points rounds prof: explanation');
  has(rPr[0], '--prof-warmup 5', 'points rounds prof: warmup linked');
  has(rPr[0], '--prof-ranks 0,3', 'points rounds prof: ranks linked');
  has(rPr[0], "-P '/mnt/xfs/prof_{L}'", 'points rounds prof: -P included');
  ok($('pointsCmdBoxNoprof').classList.contains('is-off') &&
     !$('pointsCmdBoxProf').classList.contains('is-off'),
     'points rounds prof: steady preview greyed');
  ok($('pointsCopyNoprof').disabled && !$('pointsCopyProf').disabled,
     'points rounds prof: only trace copy enabled');
  w.__copied = '';
  clickId('pointsCopyProf');
  ok(String(w.__copied).includes('--prof-only') &&
     String(w.__copied).includes('--prof-warmup 5') &&
     String(w.__copied).includes('--prof-ranks 0,3') &&
     String(w.__copied).includes("-P '/mnt/xfs/prof_{L}'"),
     'points rounds prof: trace preview copies');
  w.__copied = '';
  clickId('pointsCopyNoprof');
  ok(w.__copied === '', 'points rounds prof: disabled preview cannot copy');

  // prof-ranks 只在包含 prof 轮时校验
  setAndFire('pointsProfRanks', 'abc');
  has(msgs(), '--prof-ranks 格式不对', 'points rounds prof: invalid ranks blocked');
  ok($('pointsCopyProf').disabled, 'points rounds prof: invalid ranks disables copy');
  setAndFire('pointsRounds', 'noprof', 'change');
  ok(!E().length, 'points rounds noprof: invalid ranks ignored');
  ok(!$('pointsCopyNoprof').disabled, 'points rounds noprof: copy stays enabled');
  setAndFire('pointsProfRanks', 'all');
  setAndFire('pointsRounds', 'prof', 'change');

  // URL 往返：轮次随分享链接恢复
  ok(w.BUILDER.getState().points.pointsRounds === 'prof',
     'points rounds: state exports selection');
  const roundsUrl = w.BUILDER.shareUrl();
  w.BUILDER.applyAllDefaults();
  w.location.hash = roundsUrl.split('#')[1];
  ok(w.BUILDER.loadFromHash(), 'points rounds: share hash loads');
  w.refresh();
  ok($('pointsRounds').value === 'prof', 'points rounds: URL round-trip');
  const rUrlRuns = pointRuns(selectTab('cmd'));
  ok(rUrlRuns.length === 1 && rUrlRuns[0].includes('--prof-only'),
     'points rounds: URL restores script too');

  setAndFire('pointsRounds', 'both', 'change');
  const rBoth = pointRuns(selectTab('cmd'));
  ok(rBoth.length === 2 && rBoth[0].includes('--noprof-only') &&
     rBoth[1].includes('--prof-only'), 'points rounds both: two commands restored');
  ok(!$('pointsCmdBoxNoprof').classList.contains('is-off') &&
     !$('pointsCmdBoxProf').classList.contains('is-off'),
     'points rounds both: both previews active');
  ok(!$('pointsCopyNoprof').disabled && !$('pointsCopyProf').disabled,
     'points rounds both: both previews copyable');

  // 轮次与投递方式 / MoE 后端互不影响
  setAndFire('pointsRounds', 'prof', 'change');
  setAndFire('pointsDelivery', 'spec', 'change');
  setAndFire('pointsMoemode', 'deepgemm', 'change');
  const crossRun = pointRuns(selectTab('cmd'));
  ok(crossRun.length === 1 && (crossRun[0].includes('--points-spec')
       || crossRun[0].includes('--points-json')) &&
     crossRun[0].includes('--prof-only') &&
     crossRun[0].includes('--moemode deepgemm'),
     'points rounds: independent of delivery/moemode');
  setAndFire('pointsDelivery', 'file', 'change');
  setAndFire('pointsRounds', 'both', 'change');
  setAndFire('pointsProfDir', '');
  setAndFire('pointsWarmup', '2');
  setAndFire('pointsProfRanks', 'all');

  // MoE 后端：默认 deepgemm（2026-10-21 用户口径）也显式写；切 ascendc 后两条命令都带
  ok($('pointsMoemode').value === 'deepgemm', 'points moemode: default deepgemm');
  ok($('realMoemode').value === 'deepgemm', 'real moemode: default deepgemm');
  ok($('moeBackend').value === 'deepgemm', 'launch moemode: default deepgemm');
  ok($('pointsMoemodeNote').classList.contains('hidden'),
     'points moemode: note hidden on deepgemm');
  const runsDg = pointRuns(selectTab('cmd'));
  ok(runsDg.length === 2 && runsDg.every(l => l.includes('--moemode deepgemm')),
     'points script: deepgemm explicit on both commands');
  ok(runsDg.every(l =>
    l.indexOf('--console') > 0 &&
    l.indexOf('--console') < l.indexOf('--moemode deepgemm') &&
    l.indexOf('--moemode deepgemm') < l.indexOf(' -o ')),
    'points script: moemode order (console -> moemode -> -o)');
  setAndFire('pointsMoemode', 'ascendc', 'change');
  ok(!$('pointsMoemodeNote').classList.contains('hidden'),
     'points moemode: ascendc note shown');
  const runsAsc = pointRuns(selectTab('cmd'));
  ok(runsAsc.length === 2 && runsAsc.every(l => l.includes('--moemode ascendc')),
     'points script: ascendc on both commands');
  ok(runsAsc.every(l => !l.includes('--moemode deepgemm')),
     'points script: no stale deepgemm after switch');
  setAndFire('pointsMoemode', 'deepgemm', 'change');
  const mstate = w.BUILDER.getState();
  ok(mstate.points.pointsMoemode === 'deepgemm', 'points state: moemode exported');
  const murl = w.BUILDER.shareUrl();
  w.BUILDER.applyAllDefaults();
  w.location.hash = murl.split('#')[1];
  ok(w.BUILDER.loadFromHash(), 'points share: moemode hash loads');
  w.refresh();
  ok($('pointsMoemode').value === 'deepgemm', 'points share: moemode round-trip');
  ok($('pointsMoemodeNote').classList.contains('hidden'),
     'points share: deepgemm note restored');

  // 点位传递方式：文件 / 内联简写 / 内联 JSON
  resetAll(); w.setEntry('points'); w.refresh();
  ok($('pointsDelivery'), 'points delivery: selector exists');
  ok($('pointsDelivery').value === 'spec', 'points delivery: default is inline spec');
  ok($('pointsPathField').classList.contains('hidden'),
     'points delivery: path hidden by default (inline spec)');
  setAndFire('pointsDelivery', 'file', 'change');
  w.refresh();
  ok(!$('pointsPathField').classList.contains('hidden'),
     'points delivery: path visible in file mode');
  d.querySelector('#pointsPresets button[data-points-preset="hit80k16"]')
    .dispatchEvent(new w.Event('click', { bubbles: true }));
  w.refresh();
  let dRuns = pointRuns(selectTab('cmd'));
  ok(dRuns.length === 2 && dRuns.every(l => /--points \S+/.test(l)),
     'points delivery file: both commands use --points <path>');
  ok(dRuns.every(l => !l.includes('--points-spec') && !l.includes('--points-json')),
     'points delivery file: no inline flags');
  has($('pointsArgPreview').textContent, '--points ', 'points delivery file: arg preview');

  setAndFire('pointsDelivery', 'spec', 'change');
  ok($('pointsPathField').classList.contains('hidden'), 'points delivery spec: path hidden');
  has($('pointsDeliveryHint').textContent, '内联简写', 'points delivery spec: inline hint');
  has($('pointsArgPreview').textContent, '--points-json', 'points delivery spec: arg preview (multi-batch → json)');
  ok(!$('pointsDlJson').disabled, 'points delivery spec: json download still available');
  const specRuns = pointRuns(selectTab('cmd'));
  ok(specRuns.length === 2 && specRuns.every(l => l.includes("--points-json '")),
     'points delivery spec: both commands carry --points-json (multi-batch)');
  ok(specRuns.every(l => !l.includes('--points ./')),
     'points delivery spec: no --points file argument');
  // hit80k16 现在是「8 个点位 × 每点 1–4 批」→ spec 表达不了多批，自动回退内联 JSON
  ok(specRuns.length === 2 && specRuns.every(l => l.includes("--points-json '")),
     'points delivery spec: multi-batch falls back to inline json');
  has($('pointsArgPreview').textContent, '--points-json', 'points delivery spec: fallback preview');

  // 单批点位 → spec 生效
  setVal('pointsImportText',
    '[{"tag":"a","batches":[{"L":81920,"hit":0.96}]},' +
    ' {"tag":"b","batches":[{"L":16384,"hit":0.97}]}]');
  clickId('pointsImportPaste');
  w.refresh();
  const specRuns2 = pointRuns(selectTab('cmd'));
  const spec = (specRuns2[0].match(/--points-spec '([^']*)'/) || [])[1] || '';
  ok(spec === '81920@0.96/r1 16384@0.97/r1', 'points delivery spec: single-batch points (+/r1)');
  ok(!/\s{2,}/.test(spec), 'points delivery spec: single spaces between points');
  ok(specRuns2[1].indexOf(spec) >= 0, 'points delivery spec: same payload on prof command');

  // 校验错误仍然阻断复制/下载
  ptSet(1, '.pt-l', '65535');
  ok($('pointsCopyNoprof').disabled && $('pointsCopyProf').disabled &&
     $('pointsCopyJson').disabled && $('pointsDlJson').disabled,
     'points delivery spec: validation errors still block copy/download');
  ptSet(1, '.pt-l', '16k');

  setAndFire('pointsDelivery', 'json', 'change');
  ok($('pointsPathField').classList.contains('hidden'), 'points delivery json: path hidden');
  const jsonRuns = pointRuns(selectTab('cmd'));
  ok(jsonRuns.length === 2 && jsonRuns.every(l => l.includes("--points-json '")),
     'points delivery json: both commands carry --points-json');
  has(jsonRuns[0], "--points-json '[{", 'points delivery json: single-quoted payload');
  const rawJson = (jsonRuns[0].match(/--points-json '([^']*)'/) || [])[1] || '';
  let parsedJson = null;
  try { parsedJson = JSON.parse(rawJson); } catch (e) { /* 下面断言报错 */ }
  ok(Array.isArray(parsedJson) && parsedJson.length === 2,
     'points delivery json: parseable 2 points');
  ok(parsedJson && parsedJson.every(p => p.tag && Array.isArray(p.batches)
      && p.batches.length === 1 && typeof p.batches[0].L === 'number'),
     'points delivery json: v2 shape (tag + batches[{L,hit}])');
  ok(rawJson === JSON.stringify(JSON.parse($('pointsJsonPreview').textContent)),
     'points delivery json: compact and equals table preview');
  ok(!/\s/.test(rawJson), 'points delivery json: no spaces');
  ok(parsedJson && parsedJson.every(p => Object.keys(p).every(k =>
      ['tag', 'batches', 'replay'].includes(k))),
     'points delivery json: only v2 keys (+ per-point replay)');
  const dstate = w.BUILDER.getState();
  ok(dstate.points.pointsDelivery === 'json', 'points state: delivery exported');
  const durl = w.BUILDER.shareUrl();
  w.BUILDER.applyAllDefaults();
  w.location.hash = durl.split('#')[1];
  ok(w.BUILDER.loadFromHash(), 'points share: delivery hash loads');
  w.refresh();
  ok($('pointsDelivery').value === 'json', 'points share: delivery round-trip');
  ok($('pointsPathField').classList.contains('hidden'), 'points share: inline path hidden');
  setAndFire('pointsDelivery', 'file', 'change');
  ok(!$('pointsPathField').classList.contains('hidden'),
     'points delivery: switching back shows path');
  ok(pointRuns(selectTab('cmd')).every(l => /--points \S+/.test(l)),
     'points delivery: file mode restored');

  // 预览复制按钮：复制的是单条完整命令
  clickId('pointsCopyNoprof');
  ok(String(w.__copied).includes('--noprof-only') && !String(w.__copied).includes('--prof-only'),
     'points copy: noprof command');
  clickId('pointsCopyProf');
  ok(String(w.__copied).includes('--prof-only') &&
     String(w.__copied).includes('--prof-warmup 2') &&
     String(w.__copied).includes('--prof-ranks all'),
     'points copy: prof command');
  clickId('pointsCopyJson');
  ok(String(w.__copied).trim().startsWith('['), 'points copy: json');

  resident();

  /* ══════════ 14) 使用说明 / 复制 / 下载 / 分享 ══════════ */
  const usage = d.querySelector('details.usage');
  ok(usage, 'usage: details.usage exists');
  const usageText = $('usageContent').textContent;
  has(usageText, '三步', 'usage: three steps');
  has(usageText, '常见错误', 'usage: common errors');
  has(usageText, 'npu-smi', 'usage: npu-smi');
  has(usageText, 'tp8 增量已验证', 'usage: tp8 verified');
  has(usageText, '请求队列', 'usage: queue');
  has(usageText, 'dry-run', 'usage: dry-run');
  $('usageBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
  ok($('usageModal').classList.contains('open'), 'usage: modal opens');
  has($('usageBody').textContent, '三步', 'usage: modal has content');
  d.querySelector('#usageModal .xbtn').dispatchEvent(new w.Event('click', { bubbles: true }));
  ok(!$('usageModal').classList.contains('open'), 'usage: modal closes');

  /* ══════════ 15) 首次打开 / 刷新 / URL 优先级 ══════════ */
  const FRESH_URL = 'https://ds41.test/tools/bench_command_builder.html';
  function bootFresh(hash, scenario, sidebar) {
    const vcf = new VirtualConsole();
    vcf.on('jsdomError', e => { errors.push('JSDOM(fresh): ' + e.message); fail++; });
    return new JSDOM(html, {
      runScripts: 'dangerously',
      pretendToBeVisual: true,
      url: FRESH_URL + (hash || ''),
      virtualConsole: vcf,
      beforeParse(win) {
        if (scenario) win.localStorage.setItem('ds41-bench-builder-scenario', scenario);
        if (sidebar) win.localStorage.setItem('ds41-bench-builder-sidebar', sidebar);
      },
    });
  }
  const freshDefault = bootFresh('', '', '');
  const fd = freshDefault.window.document;
  ok(fd.body.dataset.scenario === 'points', 'first open: points by default');
  ok(fd.querySelector('#scenarioNav [data-nav="points"]').classList.contains('on'),
     'first open: points nav active');
  freshDefault.window.close();

  const freshStored = bootFresh('', 'real', '');
  const fs = freshStored.window.document;
  ok(fs.body.dataset.scenario === 'real', 'refresh: localStorage restores scenario');
  ok(!fs.getElementById('realCard').classList.contains('hidden'),
     'refresh: restored card visible');
  freshStored.window.close();

  const freshRail = bootFresh('', '', 'collapsed');
  const fr = freshRail.window.document;
  ok(fr.getElementById('appLayout').classList.contains('rail-collapsed'),
     'refresh: sidebar collapse restored');
  ok(fr.getElementById('sidebarToggle').getAttribute('aria-expanded') === 'false',
     'refresh: sidebar aria restored');
  freshRail.window.close();

  // URL(#cfg) 优先于 localStorage：分享链接在别的机器/上次场景不同也能开对
  w.setEntry('launch');
  w.refresh();
  const launchHash = w.BUILDER.shareUrl().split('#')[1];
  const freshUrl = bootFresh('#' + launchHash, 'real', '');
  const fu = freshUrl.window.document;
  ok(fu.body.dataset.scenario === 'launch', 'priority: URL overrides localStorage');
  ok(fu.querySelector('#scenarioNav [data-nav="launch"]').classList.contains('on'),
     'priority: URL scenario nav follows hash');
  freshUrl.window.close();

  resident();
  setPrompt(0, '复制测试');
  c = cmd();
  $('copyBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
  setTimeout(() => {
    try {
      ok(w.__copied === c, 'copy: current tab text');
      ok($('toast').classList.contains('show'), 'copy: toast');
      $('dlBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
      ok(w.__lastDownload === 'bench_all.sh', 'download: all tab name');
      selectTab('client');
      $('dlBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
      ok(w.__lastDownload === 'bench_client.sh', 'download: client tab name');
      $('shareBtn').dispatchEvent(new w.Event('click', { bubbles: true }));
      setTimeout(() => {
        try {
          has(w.__copied || '', '#cfg=', 'share: clipboard has cfg');
          w.BUILDER.setQueue([]);
          w.refresh();
          ok($('copyBtn').disabled && $('dlBtn').disabled && $('shareBtn').disabled,
             'blocked: actions disabled');
          // 日志等级：四种入口都要能设（launch/hit/real/points）
          w.setEntry('launch'); w.refresh();
          setAndFire('launchLogLevel', 'pure', 'change');
          has(cmd(), '--log-level pure', 'log-level: launch');
          setAndFire('launchLogLevel', 'info', 'change');
          lacks(cmd(), '--log-level', 'log-level: launch default omitted');
          w.setEntry('hit'); w.refresh();
          setAndFire('hitLogLevel', 'warn', 'change');
          has(cmd(), '--log-level warn', 'log-level: hit');
          setAndFire('hitLogLevel', 'info', 'change');
          w.setEntry('real'); w.refresh();
          setAndFire('realLogLevel', 'debug', 'change');
          has(cmd(), '--log-level debug', 'log-level: real');
          setAndFire('realLogLevel', 'info', 'change');
          w.setEntry('points'); w.refresh();
          has($('pointsLogLevel') ? 'y' : '', 'y', 'log-level: points control exists');
          setAndFire('pointsLogLevel', 'pure', 'change');
          const _ll = w.BUILDER.shareUrl();
          setAndFire('pointsLogLevel', 'info', 'change');
          w.BUILDER.applyAllDefaults();
          w.location.hash = _ll.split('#')[1];
          ok(w.BUILDER.loadFromHash(), 'log-level: hash loads');
          w.refresh();
          ok($('pointsLogLevel').value === 'pure', 'log-level: share round-trip');
          checkAndFire('pointsReplay', false);
          const _rp = w.BUILDER.shareUrl();
          w.BUILDER.applyAllDefaults();
          w.location.hash = _rp.split('#')[1];
          ok(w.BUILDER.loadFromHash(), 'replay: hash loads');
          w.refresh();
          ok(!$('pointsReplay').checked, 'replay: share round-trip');
          checkAndFire('pointsReplay', true);

          // 生成的脚本语法：每种场景/标签页都用 bash -n 过一遍
          (function syntaxCheckAll() {
            const cases = [];
            const grab = (label) => {
              const text = $('cmd').textContent || '';
              // 只检查 bash 标签页（points 场景还有一个 JSON 预览页）
              if (text.includes('#!/usr/bin/env bash')) cases.push([label, text]);
            };
            for (const sc of ['points', 'launch', 'hit', 'real']) {
              w.setEntry(sc); w.refresh();
              const ids = tabIds();
              for (const t of (ids.length ? ids : ['cmd'])) {
                const b = d.querySelector(`#tabs .tab[data-tab="${t}"]`);
                if (b) b.dispatchEvent(new w.Event('click', { bubbles: true }));
                grab(sc + ':' + t);
              }
            }
            w.setEntry('resident'); w.refresh();
            for (const t of tabIds()) {
              const b = d.querySelector(`#tabs .tab[data-tab="${t}"]`);
              if (b) b.dispatchEvent(new w.Event('click', { bubbles: true }));
              grab('resident:' + t);
            }
            let okAll = cases.length >= 4;
            for (const [label, text] of cases) {
              const r = spawnSync('bash', ['-n'], { input: text, encoding: 'utf8' });
              if (r.status !== 0) {
                okAll = false;
                errors.push('SYNTAX(' + label + '): ' + (r.stderr || '').slice(0, 200));
              }
            }
            ok(okAll, 'scripts: bash -n passes for all tabs (' + cases.length + ')');
            w.setEntry('points'); w.refresh();
          })();

          // 新手快速上手：3 步 + 冒烟预设
          ok(!!$('quickStart'), 'onboarding: quick start exists');
          ok($('quickStart').hasAttribute('open'), 'onboarding: open on first visit');
          has($('quickStart').textContent, '第一次用', 'onboarding: beginner wording');
          has($('quickStart').textContent, 'results.tsv', 'onboarding: tells where results are');
          ok(!!d.querySelector('#pointsPresets button[data-points-preset="smoke"]'),
             'onboarding: smoke preset button exists');
          clickId('qsSmoke');
          ok(ptRows().length === 1 && ptRows()[0].querySelector('.pt-l').value === '8192',
             'onboarding: smoke sets one 8k point');
          ok(ptRows()[0].querySelector('.pt-h').value === '0.96', 'onboarding: smoke hit');
          ok($('pointsIters').value === '1' && $('pointsRounds').value === 'noprof',
             'onboarding: smoke uses 1 iter / noprof only');
          ok($('pointsPath').value === './points_smoke8k.json', 'onboarding: smoke path');
          const qsc = selectTab('cmd');
          has(qsc, '--moemode deepgemm', 'onboarding: smoke command is runnable');
          clickId('qsHide');
          ok(!$('quickStart').hasAttribute('open'), 'onboarding: dismiss closes');
          ok(w.localStorage.getItem('ds41_qs_closed') === '1', 'onboarding: dismissal remembered');

          console.log(`\n${pass} passed, ${fail} failed`);
          if (errors.length) console.log(errors.join('\n'));
          process.exit(fail ? 1 : 0);
        } catch (e) { fail++; console.log('FATAL(inner2): ' + (e && e.stack || e)); process.exit(2); }
      }, 80);
    } catch (e) { fail++; console.log('FATAL(inner): ' + (e && e.stack || e)); process.exit(2); }
  }, 80);
} catch (e) {
  console.log('FATAL TEST ERROR: ' + (e && e.stack || e));
  process.exit(2);
}
}, 120);
