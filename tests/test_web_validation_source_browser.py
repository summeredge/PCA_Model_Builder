import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_model_results as ui


def test_independent_validation_upload_in_real_browser(tmp_path, monkeypatch):
    if not all((shutil.which("node"), os.environ.get("WEB_GEOMETRY_PLAYWRIGHT"), os.environ.get("WEB_GEOMETRY_BROWSER"))):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER")
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(42)
    history = pd.DataFrame(rng.normal(size=(180, 3)), columns=["A", "B", "C"])
    history.insert(0, "time", pd.date_range("2026-01-01", periods=180, freq="5min"))
    uploaded = web.save_upload("training.csv", history.to_csv(index=False).encode("utf-8-sig"))
    payload = dict(file_id=uploaded["file_id"], timestamp_column="time", tags=["A", "B", "C"],
                   normal_start=history.time.iloc[0].isoformat(), normal_end=history.time.iloc[-1].isoformat(),
                   sample_interval_minutes=5, filter_method="none", max_lag_minutes=10,
                   lag_step_minutes=5, n_components=2, model_purpose="normal_state", model_name="independent")
    trained = ui.train_payload(payload)
    inspection = web.inspect_payload(payload)
    model_path = web.RUNS_DIR / trained["run_id"] / "model.pcamodel"
    model_bytes = model_path.read_bytes()
    future = history.rename(columns={"time": "later_time"}).copy()
    future["later_time"] += pd.Timedelta(days=1)
    future["label"] = "后续文件"
    future.loc[90:, "A"] += 10
    future_path = tmp_path / "independent.csv"
    future.to_csv(future_path, index=False, encoding="utf-8-sig")
    windows = [dict(id="normal", type="normal_validation", start=future.later_time.iloc[2].isoformat(),
                    end=future.later_time.iloc[89].isoformat()),
               dict(id="abnormal", type="known_abnormal", start=future.later_time.iloc[90].isoformat(),
                    end=future.later_time.iloc[-1].isoformat())]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(dict(uploaded=uploaded, trained=trained, inspection=inspection,
                                      windows=windows, file=str(future_path))), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT),assert=require('node:assert/strict'),fs=require('node:fs');
const f=JSON.parse(fs.readFileSync(process.env.VALIDATION_FIXTURE,'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST')requests.push(r)});
 try {
  await page.goto(process.env.VALIDATION_URL);
  await page.evaluate(f=>{
   state.fileId=f.uploaded.file_id;state.fileName='training.csv';state.runId=f.trained.run_id;
   state.training=f.trained;state.inspection=f.inspection;state.selectedModelTags=new Set(['A','B','C']);
   state.registry=Object.fromEntries(['A','B','C'].map(t=>[t,emptyTagConfig()]));
   state.trainingWindows=f.trained.training_window_summary;
   state.validationInvestigationModel={training:state.training,tags:['A','B','C'],tagConfigs:state.registry};
   fillSelect(el('timestampColumn'),['time']);fillSelect(el('trendTags'),['A','B','C']);
   renderTraining(f.trained);resetValidationSource();showWorkflowStage('validationPanel');
  },f);
  const snapshot=()=>page.evaluate(()=>JSON.stringify([state.fileId,state.inspection,state.training,state.runId,[...state.selectedModelTags],state.trainingWindows,state.registry]));
  const before=await snapshot();assert.equal(await page.locator('#validationSource').inputValue(),'current');
  await page.locator('#validationSource').selectOption('upload');
  assert(await page.locator('#validateButton').isDisabled());
  await page.locator('#validationFileInput').setInputFiles(f.file);
  await page.locator('#validationUploadButton').click();
  await page.waitForFunction(()=>state.validationUpload?.file_id);
  await page.locator('#validationTimestampColumn').selectOption('later_time');
  await page.locator('#validationInspectButton').click();
  await page.waitForFunction(()=>state.validationUpload?.inspection);
  assert.equal(await snapshot(),before);
  assert((await page.locator('#validationSourceStatus').innerText()).includes('independent.csv'));
  await page.locator('#labelColumn').selectOption('label');
  for(const w of f.windows){
   await page.locator('#validationType').selectOption(w.type);
   await page.locator('#validationStart').fill(w.start);await page.locator('#validationEnd').fill(w.end);
   await page.locator('#addValidationWindow').click();
  }
  await page.locator('#validateButton').click();await page.waitForFunction(()=>state.validation?.scored_rows>0);
  const source=await page.evaluate(()=>state.validationUpload.file_id);
  const report=await page.evaluate(()=>state.validation);
  assert.equal(report.run_id,f.trained.run_id);assert.equal(report.validation_source.file_id,source);
  assert.deepEqual(Object.keys(report.status_by_engineering_label),['后续文件']);
  assert(report.scores.length>0&&report.contributions.length>0&&report.normal_validation_complete&&report.known_abnormal_complete);
  assert.equal(await snapshot(),before);
  await page.locator('[data-validation-investigation-window]').first().getByRole('button',{name:'查看该窗口趋势'}).click();
  await page.waitForFunction(()=>document.querySelector('#dpTrendChart')?.data?.length===3);
  const trends=requests.filter(r=>r.url().endsWith('/api/trend'));
  assert(trends.length>0&&trends.every(r=>r.postDataJSON().file_id===source&&r.postDataJSON().timestamp_column==='later_time'));
  await page.getByRole('button',{name:'返回验证结果'}).click();
  await page.locator('#validationDecision').selectOption('failed');await page.locator('#recordValidationDecision').click();
  await page.waitForFunction(()=>state.validation?.engineer_decision?.decision==='failed');
  for(const width of [1440,900,390]){
   await page.setViewportSize({width,height:1000});await page.evaluate(()=>document.fonts.ready);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
   for(const id of ['validationSource','validationFileInput','validationTimestampColumn','validationEncoding']){
    assert(await page.locator('#'+id).isVisible());
    assert(await page.locator('#'+id).evaluate(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.left>=0&&r.right<=innerWidth+1}));
   }
   if(process.env.SCREENING_ARTIFACT_DIR)await page.screenshot({path:require('node:path').join(process.env.SCREENING_ARTIFACT_DIR,`validation-source-${width}.png`),fullPage:true});
  }
  await page.locator('#validationSource').selectOption('current');
  assert.equal(await page.evaluate(()=>state.validation),null);
  assert.equal(await page.locator('#validationWindowTable tr').count(),0);
  assert(await page.locator('#recordValidationDecision').isDisabled());assert.equal(await snapshot(),before);
  assert.deepEqual(errors,[]);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True,
                                encoding="utf-8", timeout=90, env={**os.environ,
                                "VALIDATION_URL": f"http://127.0.0.1:{server.server_port}",
                                "VALIDATION_FIXTURE": str(fixture)})
        assert result.returncode == 0, result.stderr
        assert model_path.read_bytes() == model_bytes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
