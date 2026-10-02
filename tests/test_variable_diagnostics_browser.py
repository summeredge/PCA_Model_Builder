"""Opt-in browser check using the same runtime as test_web_geometry.py."""

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


def test_variable_diagnostics_fifty_tags_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    rng = np.random.default_rng(42)
    x = rng.normal(size=180)
    tags = ["FIC400001.SV_" + "LONG_TAG_" * 30, "FIC400002.SV", *[f"TAG_{i}" for i in range(2, 50)]]
    frame = pd.DataFrame({tag: (x * (i + 1) + rng.normal(scale=0.01, size=180) if i < 40 else rng.normal(size=180)) for i, tag in enumerate(tags)})
    frame[tags[48]] = rng.integers(0, 2, size=180)
    frame[tags[49]] = 1000 + rng.normal(scale=1e-4, size=180)
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=180, freq="5min"))
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],reports=[];page.on('pageerror',error=>errors.push(error.message));
  try {
    await page.goto(process.env.DIAGNOSTICS_TEST_URL);
    await page.evaluate(()=>document.fonts.ready);
    async function action(id,endpoint) {
      const pending=page.waitForResponse(r=>r.url().endsWith(endpoint));
      await page.locator('#'+id).click();const response=await pending;
      const data=await response.json();assert(response.ok(),JSON.stringify(data));return data;
    }
    await page.locator('#fileInput').setInputFiles(process.env.DIAGNOSTICS_TEST_CSV);
    await action('uploadButton','/api/upload');
    await page.locator('#timestampColumn').selectOption('time');
    await action('inspectButton','/api/inspect');
    await page.evaluate(()=>{showWorkflowStage('candidatePanel');showCandidateTool('stateExplorationPanel');});
    await page.locator('#maxLag').fill('0');
    await page.locator('#explorationClusterCount').fill('2');
    const data=await action('stateExplorationButton','/api/state-exploration/run');
    await page.waitForFunction(()=>state.exploration?.variable_diagnostics?.summary.tag_count===50);
    const diagnostics=data.variable_diagnostics;
    assert.equal(diagnostics.sample_count,data.full_point_count);
    assert.equal(diagnostics.cluster_features.length,50);
    assert(diagnostics.high_correlation_pairs.length>20);
    assert(diagnostics.tag_profiles.some(item=>item.flags.includes('近似无变化')));
    assert(diagnostics.tag_profiles.some(item=>item.flags.includes('低唯一值，疑似离散状态量')));
    assert.equal(await page.locator('.variable-diagnostics button').count(),1);
    assert((await page.locator('#explorationVariableDiagnostics').innerText()).includes('高相关变量'));
    const cdp=await page.context().newCDPSession(page);
    for(const [width,dpr] of [[1920,1],[1440,1],[900,1],[390,1],[720,2]]) {
      await page.setViewportSize({width,height:1000});
      await cdp.send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:dpr,mobile:false});
      await page.locator('.variable-diagnostics').evaluate(section=>section.querySelectorAll('details').forEach(e=>e.open=true));
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const report=await page.locator('.variable-diagnostics').evaluate(section=>{
        const rect=section.getBoundingClientRect(),button=section.querySelector('button').getBoundingClientRect();
        const tables=[...section.querySelectorAll('.table-wrap')];
        return {width:innerWidth,dpr:devicePixelRatio,pageOverflowX:document.documentElement.scrollWidth-document.documentElement.clientWidth,
          sectionWidth:rect.width,sectionRight:rect.right,buttonRight:button.right,tableCount:tables.length,
          maxTableHeight:Math.max(...tables.map(e=>e.getBoundingClientRect().height)),
          localOverflow:tables.some(e=>e.scrollWidth>e.clientWidth),
          maxTagWidth:Math.max(...Array.from(section.querySelectorAll('tbody td:first-child'),e=>e.getBoundingClientRect().width)),
          tableOutside:tables.some(e=>e.getBoundingClientRect().right>rect.right+1)};
      });
      assert.equal(report.pageOverflowX,0,JSON.stringify(report));
      assert(report.sectionWidth>0 && report.sectionRight<=width+1 && report.buttonRight<=width+1,JSON.stringify(report));
      assert(report.tableCount>=5 && report.maxTableHeight<=280 && !report.tableOutside,JSON.stringify(report));
      if(width===390) {assert(report.localOverflow);assert(report.maxTagWidth<=224,JSON.stringify(report));}
      reports.push(report);
    }
    // Navigation itself preserves the evidence; editing uses existing invalidation.
    await page.locator('#diagnosticsTagConfig').click();
    assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
    assert(await page.locator('#tagSearch').evaluate(e=>e===document.activeElement));
    assert(await page.evaluate(()=>state.exploration!==null && state.selectedModelTags.size===50));
    await page.locator('#tagOptions input[type=checkbox]').first().uncheck();
    assert(await page.evaluate(()=>state.exploration===null && state.selectedModelTags.size===49));
    assert(await page.locator('#explorationContent').evaluate(e=>e.hidden));
    assert((await page.locator('#explorationEmpty').textContent()).includes('请重新运行状态探索'));
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'PASS',reports,highPairs:diagnostics.high_correlation_pairs.length}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    env = {**os.environ, "DIAGNOSTICS_TEST_URL": f"http://127.0.0.1:{server.server_port}", "DIAGNOSTICS_TEST_CSV": str(csv)}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report["status"] == "PASS"
        (tmp_path / "variable_diagnostics_geometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
