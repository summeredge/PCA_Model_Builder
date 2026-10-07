"""Opt-in browser regression for the narrow-viewport layout contract.

Covers the candidate tool panel (`#trendPanel`, restyled to `.candidate-tool-panel`)
which hosts the widest tables in the workbench. A missing `min-width:0` on the panel
-- or an unbounded single-column `auto` grid track inside it -- lets those tables
push the whole document wider than the viewport, so the page itself scrolls
horizontally instead of the table scrolling inside its own `.table-wrap`.

The assertions below pin two things at every breakpoint:
  * `documentElement.scrollWidth === documentElement.clientWidth` (no page overflow)
  * every overflowing `.table-wrap` keeps `overflow-x:auto` and stays inside the viewport
"""

import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from pca_model_builder import web_model_results as ui


def test_narrow_viewport_has_no_page_overflow_and_tables_scroll_locally() -> None:
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:390,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try {
  await page.goto(process.env.NARROW_TEST_URL);
  await page.evaluate(()=>document.fonts.ready);

  // Reveal every stage/tool panel and fill the real table containers with representative
  // wide rows, including the checkbox and status-badge columns that drive `.control`.
  await page.evaluate(()=>{
   const $=id=>document.getElementById(id);
   const rows=list=>list.map(cells=>'<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('');
   const badge=(k,t)=>'<span class="status-label '+k+'">'+t+'</span>';
   document.querySelectorAll('.panel,.candidate-tool-panel,.inner-panel').forEach(p=>p.classList.add('active'));
   ['explorationContent','clusterContent','performanceContent','modelContent','validationContent','releaseContent','modelingSnapshot']
     .forEach(id=>{const e=$(id);if(e)e.hidden=false;});
   $('explorationClusterTable').innerHTML=rows([
     ['工况组 A','120','18.4%','6','10.0 h','0.842','1.31','110','96','87.3%','52.4','2'],
     ['工况组 B','98','15.0%','4','7.5 h','1.204','1.02','95','61','64.2%','44.1','3']]);
   $('clusterTable').innerHTML=rows([
     ['工况组 01','132','21.5%','PC1 -3.21 / PC2 0.44','<button class="secondary" type="button">查看候选时段</button>'],
     ['工况组 02','88','14.3%','PC1 2.87 / PC2 -1.10','<button class="secondary" type="button">查看候选时段</button>']]);
   $('performanceConditionTable').innerHTML=rows([
     ['PERFORMANCE','>= 45 且 <= 55','421'],['TAG_B','<= 1.5','388']]);
   $('performanceTable').innerHTML=rows([
     ['2026-01-01 08:00','2026-01-01 12:30','54','<input type="checkbox">'],
     ['2026-01-02 09:15','2026-01-02 13:45','52','<input type="checkbox">']]);
   $('validationWindowTable').innerHTML=rows([
     ['正常样本验证','2026-02-01 00:00','2026-02-02 00:00','基准窗口','<button class="secondary" type="button">删除</button>'],
     ['已知异常验证','2026-02-03 06:00','2026-02-03 18:00','清洗工况','<button class="secondary" type="button">删除</button>']]);
   $('contributionTable').innerHTML=rows([
     ['异常事件 03','SPE','TAG_A','进料温度','°C','0.412','2026-02-03 06:35'],
     ['峰值 07','T2','TAG_B','循环流量','m3/h','0.287','2026-02-03 07:10']]);
   $('candidateWindows').innerHTML='<table><thead><tr><th>窗口</th><th>来源</th><th>时间范围</th><th>状态</th><th>操作</th></tr></thead><tbody>'+rows([
     ['候选 01','趋势选择','2026-01-01 08:00 至 2026-01-01 12:30',badge('pending','待决策'),'<button class="secondary" type="button">查看趋势</button><button class="secondary" type="button">确认作为训练窗口</button><button class="secondary" type="button">删除</button>'],
     ['候选 02','优选区域','2026-01-02 09:15 至 2026-01-02 13:45',badge('accepted','已接受'),'<button class="secondary" type="button">查看趋势</button><button class="secondary" type="button" disabled>确认作为训练窗口</button><button class="secondary" type="button">删除</button>']])+'</tbody></table>';
   $('trainingWindows').innerHTML='<table><thead><tr><th>参与训练</th><th>来源</th><th>开始</th><th>结束</th><th>持续时间</th><th>原始 / 有效</th><th>质量</th><th>备注</th><th>操作</th></tr></thead><tbody>'+rows([
     ['<input type="checkbox" checked>','候选 01','2026-01-01 08:00','2026-01-01 12:30','4.5 h','54 / 52',badge('usable','可用'),'基准运行','<button class="secondary" type="button">删除</button>'],
     ['<input type="checkbox">','候选 02','2026-01-02 09:15','2026-01-02 13:45','4.5 h','52 / 49',badge('review','需确认'),'负荷偏低','<button class="secondary" type="button">删除</button>']])+'</tbody></table>';
   $('trendStats').innerHTML='<table><tbody><tr><th>样本数</th><td>1200</td><th>缺失率</th><td>0.8%</td></tr><tr><th>最小值</th><td>-3.21</td><th>最大值</th><td>4.05</td></tr></tbody></table>';
  });
  await page.waitForFunction(()=>document.querySelectorAll('.table-wrap table').length>=9);

  // The panel that lost `min-width:0` is the active candidate tool panel (#trendPanel).
  assert(await page.evaluate(()=>document.querySelector('#trendPanel').classList.contains('candidate-tool-panel')));
  assert(await page.evaluate(()=>document.querySelector('#trendPanel').classList.contains('active')));

  const reports=[];
  for(const width of [390,520,760,900,1200,1440]) {
   await page.setViewportSize({width,height:900});
   await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
   const report=await page.evaluate(()=>{
    const de=document.documentElement;
    const panel=document.querySelector('#trendPanel');
    const label=w=>(w.querySelector('th,td')?.textContent||'').trim().slice(0,8);
    const wraps=[...document.querySelectorAll('.table-wrap')].filter(w=>w.querySelector('table')&&w.getClientRects().length);
    return {
     width:innerWidth,
     pageOverflow:de.scrollWidth-de.clientWidth,
     bodyOverflow:document.body.scrollWidth-de.clientWidth,
     panelWidth:Math.round(panel.getBoundingClientRect().width),
     panelRight:Math.round(panel.getBoundingClientRect().right),
     tableWraps:wraps.length,
     tables:document.querySelectorAll('.table-wrap table').length,
     escaping:wraps.filter(w=>{const r=w.getBoundingClientRect();return r.right>innerWidth+1||r.left<-1;}).map(label),
     wrongOverflow:wraps.filter(w=>w.scrollWidth>w.clientWidth+1&&getComputedStyle(w).overflowX!=='auto').map(label),
     localScroll:wraps.filter(w=>w.scrollWidth>w.clientWidth+1).map(w=>label(w)+'('+w.clientWidth+'<'+w.scrollWidth+')'),
     panelEscapes:panel.getBoundingClientRect().right>innerWidth+1,
    };
   });
   assert.equal(report.pageOverflow,0,JSON.stringify(report));
   assert(report.bodyOverflow<=0,JSON.stringify(report));
   assert.deepEqual(report.escaping,[],JSON.stringify(report));
   assert.deepEqual(report.wrongOverflow,[],JSON.stringify(report));
   assert(!report.panelEscapes,JSON.stringify(report));
   assert(report.panelWidth<=report.width,JSON.stringify(report));
   assert.equal(report.tables,9,JSON.stringify(report));
   // The wide candidate/training tables must scroll inside their own wrapper at 390px.
   if(width===390) {
    assert(report.localScroll.some(s=>s.startsWith('窗口')),JSON.stringify(report));
    assert(report.localScroll.some(s=>s.startsWith('参与训练')),JSON.stringify(report));
   }
   reports.push(report);
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({status:'PASS',reports}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    env = {**os.environ, "NARROW_TEST_URL": f"http://127.0.0.1:{server.server_port}"}
    try:
        result = subprocess.run(
            [shutil.which("node"), "-"], input=script, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
