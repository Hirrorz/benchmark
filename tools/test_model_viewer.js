/* 模型结构查看器（单文件离线）的 jsdom 回归。
 *
 *   JSDOM_PATH=/tmp/jsdom_env/node_modules/jsdom node tools/test_model_viewer.js
 *
 * 覆盖：
 *   1) 离线性：无外链字体/脚本/图片、无 <link>、无 @import；
 *   2) 内嵌 JSON：可解析、40 层、模板齐全；
 *   3) 左栏 40 层 + 层型筛选；
 *   4) 模块树：点层切换模板、点算子看接口、requires/provides/权重/形状/源码四个 tab；
 *   5) 几何预设：T 随预设改变（8192 → CP8 的 1024）；
 *   6) 层 20 的 split 视图（25 个叶子，含 boundary_cut）；
 *   7) 搜索、键透视、诊断、变体对比；
 *   8) Python/JS 形状求值器对拍（shape_vectors）。
 */
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require(process.env.JSDOM_PATH || "jsdom");

const HERE = __dirname;
const FILE = path.join(HERE, "model_viewer.html");
const html = fs.readFileSync(FILE, "utf8");

let pass = 0, fail = 0;
const errors = [];
function ok(cond, msg) { if (cond) pass++; else { fail++; errors.push("FAIL: " + msg); } }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* ── 1. 离线 / 静态约束 ─────────────────────────────────────────── */
ok(!/<link\b/i.test(html), "不应有 <link>（离线单文件）");
ok(!/(?:src|href)\s*=\s*["']https?:\/\//i.test(html), "不应有外链 src/href");
ok(!/@import/i.test(html), "不应有 CSS @import");
ok(!/https?:\/\/[^\s"'<>]+\.(?:js|css|woff2?|png|jpg|svg)/i.test(html), "不应有外部资源 URL");
ok(html.includes("MODEL-DATA:START") && html.includes("MODEL-DATA:END"), "缺少数据注入标记");

/* ── 2. 加载 ───────────────────────────────────────────────────── */
const vc = new VirtualConsole();
vc.on("jsdomError", e => { fail++; errors.push("JSDOM ERROR: " + e.message); });
vc.on("error", (...a) => { fail++; errors.push("CONSOLE ERROR: " + a.join(" ")); });

const dom = new JSDOM(html, {
  runScripts: "dangerously", pretendToBeVisual: true,
  url: "https://ds41.test/tools/model_viewer.html", virtualConsole: vc,
});
const w = dom.window, d = w.document;
const click = (el) => { ok(!!el, "元素存在"); if (el) el.click(); };
const q = (sel) => d.querySelector(sel);
const qa = (sel) => [...d.querySelectorAll(sel)];

(async () => {
  await sleep(120);
  ok(errors.length === 0, "启动无 console/jsdom 错误（" + errors.slice(0, 3).join(" | ") + "）");

  const api = w.__DS41M_VIEWER__;
  ok(!!api, "调试 API 存在");
  if (!api) return finish();

  /* ── 3. 数据完整性 ─────────────────────────────────────────── */
  ok(api.LAYERS.length === 40, `层数 = 40（实际 ${api.LAYERS.length}）`);
  ok(Object.keys(api.TEMPLATES).length >= 6, "模板数 ≥ 6");
  ok(!!api.TEMPLATES["tpl:c1a_source_split:split"], "存在 split 模板");
  ok(!!api.TEMPLATES["tpl:unfused"], "存在未融合 MoE 对照模板");
  ok((api.DATA.layers || []).every(l => l.weights && l.weights.length > 0), "每层有权重表");
  ok(qa("[data-layer]").length === 40, "左栏渲染 40 行");

  /* ── 4. 形状求值器对拍 ─────────────────────────────────────── */
  let bad = 0, badEx = "";
  for (const v of api.DATA.shape_vectors) {
    const env = api.envFor(v.preset, { idx: v.layer_idx || 0, ratio: v.ratio }, v.phase);
    const got = api.evalShape(v.expr, env);
    if (JSON.stringify(got) !== JSON.stringify(v.expected)) {
      bad++; if (!badEx) badEx = `${v.expr}@${v.preset}/${v.ratio}/${v.phase}: ${JSON.stringify(got)} != ${JSON.stringify(v.expected)}`;
    }
  }
  ok(bad === 0, `JS/Python 形状向量一致（${api.DATA.shape_vectors.length} 条，失败 ${bad}：${badEx}）`);

  /* ── 5. 层切换与模板 ───────────────────────────────────────── */
  ok(q("#tplBadge").textContent === "tpl:win", "层 0 用 tpl:win");
  click(q('[data-layer="2"]'));
  await sleep(60);
  ok(q("#tplBadge").textContent === "tpl:c2a_source", "层 2 用 tpl:c2a_source");
  ok(!!q('[data-leaf="attn.compressor.kv_compress"]'), "层 2 有 compressor");
  ok(!!q('[data-leaf="attn.indexer.search.qli"]'), "层 2 有 QLI");
  click(q('[data-layer="3"]'));
  await sleep(60);
  ok(!q('[data-leaf="attn.compressor.kv_compress"]'), "层 3（consumer）没有 compressor");

  /* ── 6. 详情：接口 / 权重 / 形状 / 源码 ────────────────────── */
  click(q('[data-layer="2"]'));
  await sleep(50);
  click(q('[data-leaf="attn.q_proj.wq_a"]'));
  await sleep(50);
  ok(q("#dTitle").textContent === "attn.q_proj.wq_a", "详情标题 = 算子路径");
  ok(/requires/.test(q("#dBody").textContent) && /provides/.test(q("#dBody").textContent),
     "接口 tab 含 requires/provides");
  ok(/model\.hidden/.test(q("#dBody").textContent), "requires 显示规范键 model.hidden");
  ok(/hidden\.x/.test(q("#dBody").textContent), "requires 显示 hidden.x 及来源");
  click(q('[data-tab="weights"]'));
  await sleep(30);
  ok(/attn\.wq_a\.weight/.test(q("#dBody").textContent), "权重 tab 含 attn.wq_a.weight");
  click(q('[data-tab="shapes"]'));
  await sleep(30);
  ok(/\[8192, 5120\]/.test(q("#dBody").textContent), "形状 tab 求值 [8192, 5120]");
  click(q('[data-tab="src"]'));
  await sleep(30);
  ok(/exec_ops\.py/.test(q("#dBody").textContent), "源码 tab 含 exec_ops.py");

  /* ── 7. 几何预设切换 ───────────────────────────────────────── */
  const ps = q("#presetSel");
  ps.value = "cp8_L8192";
  ps.dispatchEvent(new w.Event("change", { bubbles: true }));
  await sleep(50);
  click(q('[data-tab="shapes"]'));
  await sleep(30);
  ok(/\[1024, 5120\]/.test(q("#dBody").textContent), "CP8 预设下 T=1024");
  // SWA replay：层 21+ 整层只算最后 140 行
  ok([...ps.options].some(o => o.value === "prefill_replay"), "有 prefill+replay 预设");
  ps.value = "prefill_replay";
  ps.dispatchEvent(new w.Event("change", { bubbles: true }));
  await sleep(50);
  ok(/T140↺/.test(q('[data-layer="21"]').textContent), "左栏层 21 标 replay（T140↺）");
  ok(/T8192/.test(q('[data-layer="0"]').textContent), "左栏层 0 仍是 T8192");
  click(q('[data-layer="21"]')); await sleep(40);
  click(q('[data-leaf="attn.q_proj.wq_a"]')); await sleep(30);
  click(q('[data-tab="shapes"]')); await sleep(30);
  ok(/\[140, 5120\]/.test(q("#dBody").textContent), "replay 层 21 的 wq_a 形状 = [140, 5120]");
  click(q('[data-layer="0"]')); await sleep(40);
  click(q('[data-leaf="attn.q_proj.wq_a"]')); await sleep(30);
  click(q('[data-tab="shapes"]')); await sleep(30);
  ok(/\[8192, 5120\]/.test(q("#dBody").textContent), "replay 预设下层 0 仍是 [8192, 5120]");
  ps.value = "prefill";
  ps.dispatchEvent(new w.Event("change", { bubbles: true }));
  await sleep(50);

  /* ── 8. 层 20 split 视图 ───────────────────────────────────── */
  click(q('[data-layer="20"]'));
  await sleep(50);
  const vs = q("#viewSel");
  ok([...vs.options].some(o => o.value === "split"), "层 20 有 split 选项");
  vs.value = "split";
  vs.dispatchEvent(new w.Event("change", { bubbles: true }));
  await sleep(60);
  ok(q("#tplBadge").textContent === "tpl:c1a_source_split:split", "切到 split 模板");
  ok(qa("[data-leaf]").length === 25, `split 叶子数 = 25（实际 ${qa("[data-leaf]").length}）`);
  ok(!!q('[data-leaf="attn.boundary_cut"]'), "split 含 boundary_cut");
  ok(!!q('[data-leaf="attn.kv_indexer.idx_k_write"]'), "split 含 idx_k_write");
  click(q('[data-leaf="attn.boundary_cut"]'));
  await sleep(40);
  ok(/indexer\.k_ready/.test(q("#dBody").textContent), "boundary_cut 提供 indexer.k_ready");

  /* ── 8b. 未融合 MoE 对照视图 ──────────────────────────────── */
  click(q('[data-layer="0"]'));
  await sleep(30);
  const vs2 = q("#viewSel");
  ok([...vs2.options].some(o => o.value === "unfused"), "有 unfused 对照选项");
  vs2.value = "unfused";
  vs2.dispatchEvent(new w.Event("change", { bubbles: true }));
  await sleep(50);
  ok(q("#tplBadge").textContent === "tpl:unfused", "切到 unfused 模板");
  ok(!!q('[data-leaf="ffn.moe.experts.gmm"]'), "unfused 含 gmm");
  ok(!q('[data-leaf="ffn.moe.experts.mega_moe"]'), "unfused 不含 mega_moe");

  /* ── 9. 搜索 / 键透视 / 诊断 / 对比 ─────────────────────────── */
  click(q('[data-layer="2"]'));       // 回到 c2a_source（含 kv_compress）
  await sleep(40);
  const se = q("#search");
  se.value = "kv_compress";
  se.dispatchEvent(new w.Event("input", { bubbles: true }));
  await sleep(40);
  ok(qa("#tree .hit").length >= 1, "搜索 kv_compress 有命中");
  ok(qa("[data-layer]").length === 4, `搜索 kv_compress 时左栏只剩 4 层（实际 ${qa("[data-layer]").length}）`);
  se.value = "";
  se.dispatchEvent(new w.Event("input", { bubbles: true }));
  await sleep(30);
  ok(qa("[data-layer]").length === 40, "清空搜索后恢复 40 层");
  // 层型 chip 筛选
  const chip = q('[data-kind="c1a_indexsource"]');
  ok(!!chip, "有 c1a_indexsource 筛选 chip");
  if (chip) {
    chip.click(); await sleep(30);
    ok(qa("[data-layer]").length === 4, `chip 筛选后 4 层（实际 ${qa("[data-layer]").length}）`);
    chip.click(); await sleep(30);
  }

  click(q("#helpBtn"));
  await sleep(30);
  ok(!q("#modal").hidden && /模块规格/.test(q("#modalTitle").textContent), "说明弹窗打开");
  click(q("#modalClose"));
  await sleep(20);

  click(q("#diagBtn"));
  await sleep(30);
  ok(!q("#modal").hidden && q("#modalTitle").textContent === "诊断", "诊断弹窗打开");
  ok(/recipes 算子/.test(q("#modalBody").textContent), "诊断含覆盖统计");
  click(q("#modalClose"));
  await sleep(20);

  click(q("#diffBtn"));
  await sleep(40);
  ok(!q("#modal").hidden && q("#modalTitle").textContent === "变体对比", "对比弹窗打开");
  ok(/仅 /.test(q("#modalBody").textContent), "对比含增删算子");
  click(q("#modalClose"));
  await sleep(20);

  // 键透视：点击详情里的键
  click(q('[data-layer="2"]'));
  await sleep(30);
  click(q('[data-leaf="attn.q_proj.wq_a"]'));
  await sleep(40);
  click(q('[data-tab="iface"]'));
  await sleep(30);
  const keyCell = q('#dBody td.key[data-key="attn.qr"]');
  ok(!!keyCell, "详情里有可点击的键 attn.qr");
  if (keyCell) {
    keyCell.click();
    await sleep(40);
    ok(!q("#modal").hidden && /键透视/.test(q("#modalTitle").textContent), "键透视弹窗打开");
  }

  /* ── 10. 积木库 / 组装蓝图 / 模块构建器 ───────────────────── */
  click(q('[data-mode="catalog"]')); await sleep(60);
  ok(!q("#modePage").hidden && q(".layout").style.display === "none", "切到积木库模式");
  ok(qa(".plugCard").length === 41, `积木库 41 个插件（实际 ${qa(".plugCard").length}）`);
  click(qa(".plugCard")[0]); await sleep(50);
  ok(!q("#modal").hidden && /积木 ·/.test(q("#modalTitle").textContent), "插件详情弹窗");
  click(q("#modalClose")); await sleep(20);

  click(q('[data-mode="blueprint"]')); await sleep(60);
  ok(qa(".bpSection").length >= 10, `组装蓝图含配方段+模板（${qa(".bpSection").length}）`);
  ok(qa(".bpChip").length > 100, "蓝图有积木 chips");
  const bpChip = qa(".bpChip").find(el => el.dataset.node);
  ok(!!bpChip, "蓝图 chip 可跳转");
  if (bpChip) { bpChip.click(); await sleep(60);
    ok(q("#modePage").hidden && q(".layout").style.display !== "none", "从蓝图跳回结构"); }

  click(q('[data-mode="builder"]')); await sleep(60);
  ok(!!q("#bOp") && !!q("#bReq"), "构建器表单存在");
  const bReq = q("#bReq");
  bReq.value = "hidden.x=[T,H]\ncmp.latent?=[T,512]\nbogus.key";
  bReq.dispatchEvent(new w.Event("input", { bubbles: true }));
  click(q("#bValidate")); await sleep(30);
  ok(/未解析/.test(q("#bResult").textContent), "构建器报未解析键");
  bReq.value = "hidden.x=[T,H]\nattn.qr=[T,1280]";
  bReq.dispatchEvent(new w.Event("input", { bubbles: true }));
  click(q("#bValidate")); await sleep(30);
  ok(/接口满足/.test(q("#bResult").textContent), "构建器判接口满足");
  ok(/插入预览/.test(q("#bResult").textContent) && /my_op/.test(q("#bResult").textContent),
     "构建器显示插入预览（新积木高亮）");
  click(q("#bGen")); await sleep(30);
  const code = q("#bCode").textContent;
  ok(/- op: "my_op"/.test(code) && /key: "hidden.x"/.test(code),
     "生成 YAML 模块配置（不再生成 Python 类）");
  ok(!!q("#bSegment"), "构建器有配方段选择");
  ok(/- my_op/.test(q("#bRecipe").textContent), "生成 recipes.yaml 片段（- my_op）");
  click(q('[data-mode="structure"]')); await sleep(50);
  ok(q("#modePage").hidden && q(".layout").style.display !== "none", "回到结构模式");

  finish();
})();

function finish() {
  if (errors.length) {
    console.log(errors.join("\n"));
  }
  console.log(`\nmodel_viewer: ${pass} passed, ${fail} failed`);
  dom.window.close();
  process.exit(fail ? 1 : 0);
}
