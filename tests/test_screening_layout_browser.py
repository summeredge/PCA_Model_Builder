"""Opt-in screening workflow check using the existing browser runtime settings."""

import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web_model_results as ui


def test_screening_layout_and_unchanged_calculation_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    rng = np.random.default_rng(42)
    n = 480
    x = np.r_[rng.normal(-3, .2, n // 2), rng.normal(3, .2, n // 2)]
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="5min"), "TAG_A": x,
                          "TAG_B": 2 * x + rng.normal(0, .1, n), "TAG_C": rng.normal(0, 1, n),
                          "PERFORMANCE": rng.normal(50, 5, n)})
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[],reports=[];page.on('pageerror',e=>errors.push(e.message));
 try {
  await page.goto(process.env.SCREENING_TEST_URL);await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('meta[name="pca-model-builder-build"]').getAttribute('content'),process.env.SCREENING_BUILD_ID);
  async function action(id,endpoint) {
   const pending=page.waitForResponse(r=>r.url().endsWith(endpoint));
   await page.locator('#'+id).click();const response=await pending;
   const data=await response.json();assert(response.ok(),JSON.stringify(data));return data;
  }
  await page.locator('#fileInput').setInputFiles(process.env.SCREENING_TEST_CSV);
  await action('uploadButton','/api/upload');await page.locator('#timestampColumn').selectOption('time');
  await action('inspectButton','/api/inspect');
  await page.locator('[data-panel="candidatePanel"]').click();
  await page.locator('[data-panel="stateExplorationPanel"]').click();
  await page.locator('#addEligibilityKeep').click();
  const keep=page.locator('#eligibilityKeepConditions .eligibility-condition');
  await keep.locator('select').selectOption('TAG_A');await keep.locator('[data-field="minimum"]').fill('-10');
  await action('refreshEligibilitySummary','/api/modeling-eligibility');
  assert((await page.locator('#eligibilitySummary').textContent()).includes('480'));
  await page.locator('#addEligibilityExcludeGroup').click();
  const exclude=page.locator('#eligibilityExcludeGroups .eligibility-condition');
  await exclude.locator('select').selectOption('TAG_A');await exclude.locator('[data-field="minimum"]').fill('-10');
  await action('refreshEligibilitySummary','/api/modeling-eligibility');
  assert(await page.locator('#eligibilitySummary').evaluate(e=>e.classList.contains('warning')));
  await exclude.getByRole('button',{name:'删除条件',exact:true}).click();
  await keep.getByRole('button',{name:'删除条件',exact:true}).click();
  assert.equal(await page.locator('.eligibility-condition').count(),0);
  await action('refreshEligibilitySummary','/api/modeling-eligibility');
  assert(!await page.locator('#eligibilitySummary').evaluate(e=>e.classList.contains('warning')||e.classList.contains('notice')));
  const start=await page.locator('#analysisStart').inputValue(),end=await page.locator('#analysisEnd').inputValue();
  await page.locator('#analysisStart').fill('2026-01-01T01:00');await page.locator('#analysisEnd').fill('2026-01-02T12:00');
  await page.locator('#analysisStart').fill(start);await page.locator('#analysisEnd').fill(end);
  await page.locator('#sampleInterval').fill('10');await page.locator('#resamplingMethod').selectOption('mean');
  await page.locator('#filterMethod').selectOption('first_order');await page.locator('#firstOrderAlpha').fill('0.2');
  assert(await page.locator('#firstOrderAlpha').isEnabled());
  await page.locator('#filterMethod').selectOption('trailing_mean');await page.locator('#smoothingWindow').fill('15');
  assert(await page.locator('#smoothingWindow').isEnabled());
  await page.locator('#smoothingWindow').fill('10');await page.locator('#filterMethod').selectOption('none');
  await page.locator('#sampleInterval').fill('5');await page.locator('#resamplingMethod').selectOption('none');
  await page.locator('#maxLag').fill('10');await page.locator('#lagStep').fill('5');
  await page.locator('#explorationClusterCount').fill('2');await page.locator('#explorationMinimumDuration').fill('30');
  await page.locator('#explorationCandidateCount').fill('3');await page.locator('#explorationMaximumPlotPoints').fill('1200');
  await page.locator('#explorationPerformanceTag').selectOption('PERFORMANCE');
  await page.locator('#explorationPerformanceDirection').selectOption('target_range');
  await page.locator('#explorationTargetMin').fill('45');await page.locator('#explorationTargetMax').fill('55');
  const data=await action('stateExplorationButton','/api/state-exploration/run');
  await page.waitForFunction(()=>!document.querySelector('#explorationContent').hidden);
  const comparisonKeys=['exploration_config','performance_config','preprocessing_summary','exploratory_model_summary',
    'cluster_quality','variable_diagnostics','cluster_centers','full_point_count','returned_point_count','cluster_summaries','cluster_series_full','data_usage'];
  if(process.env.SCREENING_BASELINE_JSON) {
   const baseline=JSON.parse(fs.readFileSync(process.env.SCREENING_BASELINE_JSON,'utf8'));
   for(const key of comparisonKeys)assert.deepEqual(data[key],baseline[key],key);
  }
  assert.equal(await page.locator('.screening-kpis .metric').count(),4);
  assert((await page.locator('.screening-judgment').innerText()).includes(data.cluster_quality.engineering_hint.state_exploration));
  assert((await page.locator('#explorationQualityDetails').innerText()).includes('时间连续性'));
  assert(await page.locator('#explorationClusterTable tr').count()>0);
  const artifacts=process.env.SCREENING_ARTIFACT_DIR||process.env.SCREENING_TEST_DIR;
  for(const [width,zoom] of [[1920,1],[1440,1],[1200,1],[900,1],[390,1],[1440,2]]) {
   await page.setViewportSize({width,height:1000});
   await page.evaluate(zoom=>{document.body.style.zoom=zoom;},zoom);
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   const report=await page.evaluate(()=>{
    const rect=e=>e.getBoundingClientRect(),spread=a=>Math.max(...a)-Math.min(...a);
    const fields=[...document.querySelectorAll('#modelingEligibility input,#candidatePanel .shared-preprocessing input,#candidatePanel .shared-preprocessing select,#stateExplorationPanel .exploration-controls input,#stateExplorationPanel .exploration-controls select')].filter(e=>e.getClientRects().length);
    const controls=fields.map(rect),kpis=[...document.querySelectorAll('.screening-kpis .metric')].map(rect);
    const dates=['analysisStart','analysisEnd'].map(id=>rect(document.getElementById(id)));
    return {width:innerWidth,zoom:document.body.style.zoom,
     configHeight:rect(document.getElementById('stateExplorationButton')).bottom-rect(document.getElementById('modelingEligibility')).top,
     overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
     kpiWidthSpread:spread(kpis.map(r=>r.width)),kpiHeightSpread:spread(kpis.map(r=>r.height)),dateWidthSpread:spread(dates.map(r=>r.width)),
     clipped:fields.filter(e=>{const r=rect(e),p=rect(e.parentElement);return r.left<p.left-1||r.right>p.right+1||r.width<100;}).map(e=>e.id),
     overlap:controls.some((a,i)=>controls.slice(i+1).some(b=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1))};
   });
   assert.equal(report.overflow,0,JSON.stringify(report));assert(!report.overlap,JSON.stringify(report));assert.deepEqual(report.clipped,[],JSON.stringify(report));
   assert(report.kpiWidthSpread<=1&&report.kpiHeightSpread<=1&&report.dateWidthSpread<=1,JSON.stringify(report));
   reports.push(report);
   if(width===1440&&zoom===1) {
    await page.screenshot({path:path.join(artifacts,'screening-after.png'),fullPage:true});
    await page.locator('#modelingEligibility').screenshot({path:path.join(artifacts,'screening-conditions.png')});
    await page.locator('#explorationClusterQuality').screenshot({path:path.join(artifacts,'screening-results.png')});
   }
  }
  await page.evaluate(()=>{document.body.style.zoom=1;});
  await page.locator('#diagnosticsTagConfig').click();
  assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
  await page.locator('[data-panel="candidatePanel"]').click();
  await page.locator('#explorationClusterCount').fill('3');await page.locator('#explorationClusterCount').blur();
  assert(await page.locator('#explorationContent').evaluate(e=>e.hidden));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(artifacts,'screening-geometry.json'),JSON.stringify(reports,null,2));
  console.log(JSON.stringify({status:'PASS',reports,baselineCompared:Boolean(process.env.SCREENING_BASELINE_JSON)}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    env = {**os.environ, "SCREENING_TEST_URL": f"http://127.0.0.1:{server.server_port}",
           "SCREENING_TEST_CSV": str(csv), "SCREENING_TEST_DIR": str(tmp_path), "SCREENING_BUILD_ID": ui.WEB_BUILD_ID}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
