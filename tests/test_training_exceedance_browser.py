"""Real training/API/Plotly integration plus controlled response-order regressions."""

import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_model_results as ui, training_exceedance


def test_training_exceedance_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "MAX_CHART_POINTS", 7)
    rng = np.random.default_rng(71)
    times = pd.date_range("2026-01-01", periods=200, freq="5min").delete(slice(90, 105))
    frame = pd.DataFrame({"time": times, **{tag: rng.normal(size=len(times)) for tag in "ABC"},
                          "Load": np.linspace(10, 100, len(times))})
    frame.loc[125, "B"] = np.nan
    frame.loc[125, "Load"] = np.nan
    uploaded = web.save_upload("browser.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    request = {"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": list("ABC"),
               "normal_start": times[0].isoformat(), "normal_end": times[-1].isoformat(),
               "sample_interval_minutes": 5, "smoothing_window_minutes": 0, "filter_method": "none",
               "max_lag_minutes": 5, "lag_step_minutes": 5, "n_components": 2,
               "model_name": "browser-diagnosis", "tag_configs": {"A": {"unit": "degC"},
                   "B": {"unit": "bar"}, "C": {"unit": "m3/h"}, "Load": {"unit": "t/h"}}}
    trained = ui.train_payload(request)
    overview = training_exceedance.training_exceedance_payload({"run_id": trained["run_id"]})
    normal = next(row for row in overview["scores"] if row["t2"] < overview["t2_limits"]["95"] and row["spe"] < overview["q_limits"]["95"])
    high = [i for i, row in enumerate(overview["scores"])
            if row["spe"] is not None and row["spe"] >= overview["q_limits"]["95"]]
    low = [i for i, row in enumerate(overview["scores"])
           if row["spe"] is not None and row["spe"] < overview["q_limits"]["95"]]
    assert len(high) >= 2 and len(low) >= 2
    pairs = next(((high_pair, low_pair)
                  for high_pair in ((high[0], high[1]), (high[-2], high[-1]))
                  for low_pair in ((low[0], low[1]), (low[-2], low[-1]))
                  if high_pair[1] < low_pair[0] or low_pair[1] < high_pair[0]), None)
    assert pairs is not None
    anomaly_pair, reference_pair = pairs
    score_rows = overview["scores"]
    dual = {"run_id": trained["run_id"], "action": "compare", "load_tag": "Load",
            "anomaly_start": score_rows[anomaly_pair[0]]["timestamp"],
            "anomaly_end": score_rows[anomaly_pair[1]]["timestamp"],
            "reference_start": score_rows[reference_pair[0]]["timestamp"],
            "reference_end": score_rows[reference_pair[1]]["timestamp"]}
    comparison = training_exceedance.training_exceedance_payload(dual)
    fixture = tmp_path / "training.json"
    fixture.write_text(json.dumps({"trained": trained, "file_id": uploaded["file_id"], "overview": overview,
                                  "normal": normal["timestamp"], "dual": dual,
                                  "comparison": comparison}), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict'), fs=require('node:fs'),path=require('node:path');
const fixture=JSON.parse(fs.readFileSync(process.env.EXCEEDANCE_FIXTURE,'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:1000},timezoneId:'Asia/Shanghai'});
  const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.url().endsWith('/api/training-exceedance'))requests.push(r.postDataJSON());});
  await page.goto(process.env.EXCEEDANCE_URL);
  await page.waitForFunction(()=>typeof renderTraining==='function' && !!globalThis.Plotly);
  await page.evaluate(f=>{
    state.fileId=f.file_id;state.runId=f.trained.run_id;state.training=f.trained;
    state.selectedModelTags=new Set(['A','B','C']);state.registry={A:{enabled:true},B:{enabled:true},C:{enabled:true}};
    renderTraining(f.trained);document.querySelector('#modelPanel').classList.add('active');
    document.querySelector('#modelContent').hidden=false;
    window.beforeDiagnosis=JSON.stringify([state.runId,state.training,state.trainingWindows,state.registry]);
  },fixture);
  await page.locator('#trainingExceedanceButton').click();
  await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent.includes('请选择'));
  assert.deepEqual(requests.map(r=>r.action),['overview']);
  assert.equal(await page.locator('#exceedanceStatistics table').count(),0);
  assert.equal(await page.evaluate(()=>document.querySelector('#exceedanceT2').data[0].y.filter(Number.isFinite).length),fixture.overview.scores.length);
  assert(fixture.overview.scores.length>fixture.trained.scores.length);
  assert.deepEqual(await page.evaluate(()=>document.querySelector('#exceedanceT2').data.slice(1).map(t=>t.y[0])),[fixture.overview.t2_limits['95'],fixture.overview.t2_limits['99']]);
  assert(await page.evaluate(()=>document.querySelector('#exceedanceT2').data[0].y.includes(null)));
  // Exercise the real Plotly pointer selection, including only x despite y height.
  const drag=await page.evaluate(()=>{const p=document.querySelector('#exceedanceT2'),b=p.getBoundingClientRect(),l=p._fullLayout;
    return {x:b.x+l.margin.l+5,y:b.y+l.margin.t+15,width:l.xaxis._length-10};});
  await page.mouse.move(drag.x,drag.y);await page.mouse.down();await page.mouse.move(drag.x+drag.width,drag.y+25,{steps:8});await page.mouse.up();
  await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent==='区间分析完成。');
  assert(requests.some(r=>r.action==='interval'));
  assert.equal(await page.evaluate(()=>document.querySelector('#exceedanceT2').layout.shapes[0].x0),await page.locator('#exceedanceStart').inputValue());
  // Exact boundaries and full-data statistics, independent of chart sampling.
  await page.locator('#exceedanceStart').fill(fixture.overview.start);
  await page.locator('#exceedanceEnd').fill(fixture.overview.end);
  await page.locator('#exceedanceApply').click();
  await page.waitForFunction(()=>!document.querySelector('#exceedanceStatus').textContent.includes('正在计算'));
  assert.equal(await page.locator('#exceedanceStatus').innerText(),'区间分析完成。');
  assert((await page.locator('#exceedanceStatistics').innerText()).includes(String(fixture.trained.training_rows)));
  const independent=await page.evaluate(async f=>await (await fetch('/api/training-exceedance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:f.trained.run_id,action:'interval',start:f.overview.start,end:f.overview.end})})).json(),fixture);
  assert.equal(await page.locator('#exceedanceStatistics tbody tr').first().locator('td').count(),2);
  for(const metric of ['t2','spe']) {
    const summary=independent.statistics[metric],text=await page.locator('#exceedanceStatistics').innerText();
    for(const field of ['mean','max','max_limit_ratio'])assert(text.includes(summary[field].toPrecision(6)));
    for(const level of [95,99])assert(text.includes(`${summary[`exceedance_${level}`].count} / ${(summary[`exceedance_${level}`].rate*100).toFixed(1)}%`));
  }
  for(const metric of ['t2','spe']) {
    await page.locator('#exceedanceMetric').selectOption(metric);
    assert.equal(await page.locator('#exceedanceContributions tbody tr').count(),independent.contributions[metric].rows.length);
    assert((await page.locator('#exceedanceContributions tbody tr').first().innerText()).includes(independent.contributions[metric].rows[0].tag));
  }
  await page.locator('#exceedanceContributions button').filter({hasText:'B'}).click();
  await page.waitForFunction(()=>document.querySelector('#exceedanceRawTitle').textContent.includes('bar'));
  await page.waitForFunction(()=>!!document.querySelector('#exceedanceRaw').data);
  assert((await page.locator('#exceedanceRawStatus').innerText()).includes('缺失'));
  assert(await page.evaluate(()=>document.querySelector('#exceedanceRaw').data[0].y.includes(null)));
  assert.equal(await page.evaluate(()=>document.querySelector('#exceedanceRaw').layout.shapes[0].x0),fixture.overview.start);
  await page.locator('#trainingExceedanceDialog').evaluate(d=>d.scrollTop=0);
  await page.screenshot({path:path.join(process.env.EXCEEDANCE_ARTIFACTS,'exceedance-desktop.png')});
  await page.locator('#exceedanceRaw').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(process.env.EXCEEDANCE_ARTIFACTS,'exceedance-raw.png')});
  const second=independent.contributions.spe.rows.find(row=>row.tag!=='B').tag;
  await page.locator('#exceedanceContributions button').filter({hasText:second}).click();
  await page.waitForFunction(tag=>document.querySelector('#exceedanceRaw').data?.[0].name===tag,second);
  assert((await page.locator('#exceedanceRawTitle').innerText()).includes(second));
  const dragSelection=async (from,to)=>{
    const points=await page.evaluate(({from,to})=>{const p=document.querySelector('#exceedanceT2'),a=p._fullLayout.xaxis,b=p.getBoundingClientRect();return {x1:b.left+a._offset+a._length*from,x2:b.left+a._offset+a._length*to,y:b.top+p._fullLayout.margin.t+18};},{from,to});
    await page.mouse.move(points.x1,points.y);await page.mouse.down();await page.mouse.move(points.x2,points.y+20,{steps:6});await page.mouse.up();
  };
  await page.locator('#exceedanceModeAnomaly').click();await dragSelection(.2,.28);
  await page.waitForFunction(()=>document.querySelector('#exceedanceAnomalyStart').value.length>0);
  await page.locator('#exceedanceModeReference').click();await dragSelection(.68,.76);
  await page.waitForFunction(()=>document.querySelector('#exceedanceReferenceStart').value.length>0);
  await page.locator('#exceedanceAnomalyStart').fill(fixture.dual.anomaly_start);
  await page.locator('#exceedanceAnomalyEnd').fill(fixture.dual.anomaly_end);
  await page.locator('#exceedanceReferenceStart').fill(fixture.dual.reference_start);
  await page.locator('#exceedanceReferenceEnd').fill(fixture.dual.reference_end);
  await page.locator('#exceedanceLoadTag').selectOption('Load');
  await page.locator('#exceedanceCompareApply').click();
  await page.waitForFunction(()=>!document.querySelector('#exceedanceCompareStatus').textContent.includes('正在计算'));
  assert.equal(await page.locator('#exceedanceCompareStatistics tbody tr').count(),12,
    JSON.stringify({status:await page.locator('#exceedanceCompareStatus').innerText(),
      anomaly:[await page.locator('#exceedanceAnomalyStart').inputValue(),await page.locator('#exceedanceAnomalyEnd').inputValue()],
      reference:[await page.locator('#exceedanceReferenceStart').inputValue(),await page.locator('#exceedanceReferenceEnd').inputValue()],dual:fixture.dual}));
  assert.equal(await page.locator('#exceedanceCompareContributions tbody tr').count(),fixture.comparison.contributions.length);
  const comparisonStats=await page.locator('#exceedanceCompareStatistics').innerText();
  assert(comparisonStats.includes('负荷可比性状态')&&comparisonStats.includes('P10–P90 主体区间重叠'));
  assert((await page.locator('#exceedanceCompareStatus').innerText()).includes('不重叠')||
    (await page.locator('#exceedanceCompareStatus').innerText()).includes('不自动判定'));
  assert.equal(await page.locator('#exceedanceStatistics table').count(),1,'first-stage interval table stays available');
  const highlights=await page.evaluate(()=>document.querySelector('#exceedanceT2').layout.shapes.map(s=>s.fillcolor));
  assert(highlights.includes('rgba(220,38,38,.18)')&&highlights.includes('rgba(37,99,235,.18)'));
  assert.equal(await page.locator('#exceedanceAnomalyStart').inputValue(),fixture.dual.anomaly_start);
  assert.equal(await page.locator('#exceedanceReferenceStart').inputValue(),fixture.dual.reference_start);
  const comparisonTag=fixture.comparison.contributions[0].tag;
  await page.locator('#exceedanceCompareContributions button').filter({hasText:comparisonTag}).click();
  await page.waitForFunction(()=>!!document.querySelector('#exceedanceCompareRaw').data);
  assert((await page.locator('#exceedanceCompareRawTitle').innerText()).includes(comparisonTag));
  assert((await page.locator('#exceedanceCompareRawTitle').innerText()).includes(({A:'degC',B:'bar',C:'m3/h'})[comparisonTag]));
  assert((await page.locator('#exceedanceCompareRawStatus').innerText()).includes(fixture.dual.anomaly_start));
  assert.equal(await page.evaluate(()=>document.querySelector('#exceedanceCompareRaw').data.length),2);
  assert.deepEqual(await page.evaluate(()=>document.querySelector('#exceedanceCompareRaw').data.map(t=>t.yaxis)),['y','y']);
  assert.equal(await page.evaluate(()=>document.querySelector('#exceedanceCompareRaw').layout.shapes.length),2);
  assert(await page.locator('#exceedanceCompareLoadTitle').innerText().then(t=>t.includes('Load')));
  assert(await page.evaluate(()=>document.querySelector('#exceedanceCompareRaw').data.every(t=>t.connectgaps===false)));
  await page.locator('#exceedanceAnomalyClear').click();
  assert.equal(await page.locator('#exceedanceAnomalyStart').inputValue(),'');
  let colors=await page.evaluate(()=>document.querySelector('#exceedanceT2').layout.shapes.map(s=>s.fillcolor));
  assert(!colors.includes('rgba(220,38,38,.18)')&&colors.includes('rgba(37,99,235,.18)'));
  await page.locator('#exceedanceReferenceClear').click();
  colors=await page.evaluate(()=>document.querySelector('#exceedanceT2').layout.shapes.map(s=>s.fillcolor));
  assert(!colors.includes('rgba(220,38,38,.18)')&&!colors.includes('rgba(37,99,235,.18)'));
  // Zoom, pan and reset share the x range without changing the analysis.
  await page.evaluate(()=>Plotly.relayout(document.querySelector('#exceedanceT2'),{'xaxis.range':['2026-01-01T02:00:00','2026-01-01T03:00:00']}));
  await page.waitForFunction(()=>document.querySelector('#exceedanceSpe').layout.xaxis.range[0]==='2026-01-01T02:00:00');
  await page.locator('#exceedanceReset').click();
  await page.waitForFunction(s=>document.querySelector('#exceedanceSpe').layout.xaxis.range[0]===s,fixture.overview.start);
  const geometry=[];
  for(const width of [1440,900,390]) {
    await page.setViewportSize({width,height:1000});
    await page.evaluate(()=>Promise.all(['exceedanceT2','exceedanceSpe','exceedanceRaw','exceedanceCompareRaw','exceedanceCompareLoadRaw'].map(id=>Plotly.Plots.resize(document.getElementById(id)))));
    const g=await page.evaluate(()=>{const d=document.querySelector('#trainingExceedanceDialog'),s=document.querySelector('#exceedanceStart').getBoundingClientRect(),e=document.querySelector('#exceedanceEnd').getBoundingClientRect();return {width:innerWidth,overflow:d.scrollWidth-d.clientWidth,pageOverflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,inputWidths:[s.width,e.width],dialogWidth:d.getBoundingClientRect().width};});
    assert(g.overflow<=1,JSON.stringify(g));assert(g.pageOverflow<=1);assert(Math.abs(g.inputWidths[0]-g.inputWidths[1])<=1);geometry.push(g);
  }
  await page.setViewportSize({width:1440,height:1000});
  // Empty physical gap and a no-exceedance single-sample interval.
  for(const [start,end,message] of [['2026-01-01T07:31:00','2026-01-01T08:30:00','没有有效'],[fixture.normal,fixture.normal,'区间分析完成']]) {
    await page.locator('#exceedanceStart').fill(start);await page.locator('#exceedanceEnd').fill(end);await page.locator('#exceedanceApply').click();
    await page.waitForFunction(m=>document.querySelector('#exceedanceStatus').textContent.includes(m),message);
    assert((await page.locator('#exceedanceContributions').innerText()).includes('没有超限'));
  }
  // Reversed bounds reject locally; microseconds and timezone are passed verbatim.
  const count=requests.length;
  await page.locator('#exceedanceStart').fill('2026-02-01T00:00:00');await page.locator('#exceedanceEnd').fill('2026-01-01T00:00:00');await page.locator('#exceedanceApply').click();
  assert.equal(requests.length,count);assert((await page.locator('#exceedanceStatus').innerText()).includes('不晚于'));
  let releases=[],controlled=false,compareCalls=0;
  await page.route('**/api/training-exceedance',async route=>{
    if(!controlled)return route.continue();
    const p=route.request().postDataJSON();
    let data;
    if(p.action==='overview')data={...fixture.overview,run_id:p.run_id};
    else if(p.action==='compare'){
      compareCalls++;data=JSON.parse(JSON.stringify(fixture.comparison));data.run_id=p.run_id;
      data.anomaly_start=p.anomaly_start;data.anomaly_end=p.anomaly_end;data.reference_start=p.reference_start;data.reference_end=p.reference_end;
      if(compareCalls===1){data.anomaly.sample_count=999;data.warnings=['OLD RESPONSE'];}
      else if(compareCalls===2){data.anomaly.load.valid_samples=1;data.load_statistics_available=false;
        data.load_comparability_status='insufficient_evidence';data.load_p10_p90_overlap=null;
        data.warnings=['负荷数据不足，不能认定同负荷；SPE 数值对照仅为描述性、未匹配负荷比较。'];}
      else {data.load_ranges_overlap=true;data.load_p10_p90_overlap=false;data.load_comparability_status='body_disjoint';
        Object.assign(data.anomaly.load,{valid_samples:2,missing_rate:0,mean:5,median:5,p10:1,p90:9,minimum:0,maximum:10});
        Object.assign(data.reference.load,{valid_samples:2,missing_rate:0,mean:14,median:14,p10:10,p90:18,minimum:9,maximum:19});
        data.warnings=['负荷范围有重叠；请核对均值、中位数及分位数，不自动判定为同负荷。','主体负荷分布不重叠/不匹配，应重新选择参考区间。'];}
    }else data={...independent,start:p.start,end:p.end};
    await new Promise(resolve=>releases.push(async()=>{await route.fulfill({json:data});resolve();}));
  });
  controlled=true;
  const precise='2026-01-01T01:00:00.123456+08:00';
  await page.locator('#exceedanceStart').fill(precise);await page.locator('#exceedanceEnd').fill(precise);await page.locator('#exceedanceApply').click();
  while(!releases.length)await new Promise(r=>setTimeout(r,10));
  assert.equal(await page.locator('#exceedanceStart').inputValue(),precise);
  await page.locator('#exceedanceStart').fill(fixture.normal);await page.locator('#exceedanceEnd').fill(fixture.normal);await page.locator('#exceedanceApply').click();
  await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent.includes('正在计算'));
  const older=releases.shift();await older();
  while(!releases.length)await new Promise(r=>setTimeout(r,10));
  await releases.shift()();
  await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent==='区间分析完成。');
  assert.equal(await page.locator('#exceedanceStart').inputValue(),fixture.normal);
  for(const [id,key] of Object.entries({exceedanceAnomalyStart:'anomaly_start',exceedanceAnomalyEnd:'anomaly_end',exceedanceReferenceStart:'reference_start',exceedanceReferenceEnd:'reference_end'}))
    await page.locator(`#${id}`).fill(fixture.dual[key]);
  await page.locator('#exceedanceLoadTag').selectOption('Load');
  await page.locator('#exceedanceCompareApply').click();while(!releases.length)await new Promise(r=>setTimeout(r,10));
  const oldCompare=releases.shift();
  const changed=`${fixture.dual.anomaly_start}.001`;
  await page.locator('#exceedanceAnomalyStart').fill(changed);await page.locator('#exceedanceCompareApply').click();
  await page.waitForFunction(()=>document.querySelector('#exceedanceCompareStatus').textContent.includes('正在计算'));
  await oldCompare();while(!releases.length)await new Promise(r=>setTimeout(r,10));await releases.shift()();
  await page.waitForTimeout(300);
  assert.equal(await page.locator('#exceedanceCompareStatistics tbody tr').count(),12,
    JSON.stringify({status:await page.locator('#exceedanceCompareStatus').innerText(),actions:requests.map(r=>r.action),compareCalls}));
  assert(!(await page.locator('#exceedanceCompareStatus').innerText()).includes('OLD RESPONSE'));
  assert.notEqual(await page.locator('#exceedanceCompareStatistics tbody tr').nth(2).locator('td').nth(1).innerText(),'999');
  assert((await page.locator('#exceedanceCompareStatus').innerText()).includes('负荷数据不足，不能认定同负荷；SPE 数值对照仅为描述性、未匹配负荷比较。'));
  const insufficientStats=await page.locator('#exceedanceCompareStatistics').innerText();
  assert(insufficientStats.includes('证据不足；不能认定同负荷')&&insufficientStats.includes('不可判断（负荷有效样本不足）'));
  await page.locator('#exceedanceCompareApply').click();while(!releases.length)await new Promise(r=>setTimeout(r,10));await releases.shift()();
  await page.waitForFunction(()=>document.querySelector('#exceedanceCompareStatus').textContent.includes('主体负荷分布不重叠'));
  const bodySeparatedStats=await page.locator('#exceedanceCompareStatistics').innerText();
  assert(bodySeparatedStats.includes('主体分布分离；请重新选择参考区间')&&bodySeparatedStats.includes('P10–P90 主体区间重叠')&&bodySeparatedStats.includes('不重叠'));
  await page.locator('#exceedanceClose').click();
  // A delayed overview cannot overwrite a reopened session.
  await page.locator('#trainingExceedanceButton').click();while(!releases.length)await new Promise(r=>setTimeout(r,10));const oldOverview=releases.shift();
  await page.keyboard.press('Escape');await page.locator('#trainingExceedanceButton').click();await oldOverview();while(!releases.length)await new Promise(r=>setTimeout(r,10));
  await releases.shift()();await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent.includes('请选择'));
  assert.equal(await page.locator('#exceedanceStatistics table').count(),0);
  await page.locator('#exceedanceClose').click();
  // A different run must not wait behind an old run's in-flight request.
  await page.locator('#trainingExceedanceButton').click();while(!releases.length)await new Promise(r=>setTimeout(r,10));const previousRun=releases.shift();
  await page.keyboard.press('Escape');
  await page.evaluate(()=>{window.originalDiagnosisTraining=state.training;state.runId='f'.repeat(32);state.training={...state.training,run_id:state.runId};});
  await page.locator('#trainingExceedanceButton').click();
  while(!releases.length)await new Promise(r=>setTimeout(r,10));
  await releases.shift()();await page.waitForFunction(()=>document.querySelector('#exceedanceStatus').textContent.includes('请选择'));
  await previousRun();assert.equal(await page.locator('#exceedanceStatistics table').count(),0);
  await page.locator('#exceedanceClose').click();
  await page.evaluate(()=>{state.training=window.originalDiagnosisTraining;state.runId=state.training.run_id;});
  assert(await page.evaluate(()=>beforeDiagnosis===JSON.stringify([state.runId,state.training,state.trainingWindows,state.registry])));
  await page.evaluate(()=>state.trainingDiagnosticInvalidated=true);
  page.once('dialog',d=>d.dismiss());await page.locator('#trainingExceedanceButton').click();assert.equal(await page.locator('#trainingExceedanceDialog').evaluate(d=>d.open),false);
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(process.env.EXCEEDANCE_ARTIFACTS,'exceedance-geometry.json'),JSON.stringify(geometry,null,2));
  console.log(JSON.stringify({status:'PASS',trainingRows:fixture.trained.training_rows,overviewRows:fixture.overview.scores.length,geometry}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    env = {**os.environ, "EXCEEDANCE_URL": f"http://127.0.0.1:{server.server_port}",
           "EXCEEDANCE_FIXTURE": str(fixture), "EXCEEDANCE_ARTIFACTS": str(tmp_path)}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=100)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
