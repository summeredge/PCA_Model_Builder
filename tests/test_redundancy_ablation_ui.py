"""Final assembled page and opt-in real API/browser redundancy workflow."""

import hashlib
import json
import os
import shutil
import subprocess
import threading
from collections import Counter
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web_model_results as ui


def test_ablation_results_render_training_alignment_and_independent_channel_counts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for WebUI regression tests")
    source = Path(ui.__file__).with_name("model_results.js").read_text(encoding="utf-8")

    def slice_between(start: str, end: str) -> str:
        begin = source.index(start)
        return source[begin:source.index(end, begin)]

    helpers = "\n".join((
        slice_between("  const ablationPercent", "  function ablationKey()"),
        slice_between("  function ablationTable(", "  function clearAblationSchemes()"),
        slice_between("  function ablationValidationText(", "  function renderAblationSchemes("),
        slice_between("  function renderAblationSchemes(", '  el("ablationApplyButton")'),
    ))
    script = r'''
const assert=require('node:assert/strict');
class Node {}
class Element extends Node {
  constructor(){super();this.children=[];this.textContent='';this.attrs={};}
  append(...nodes){this.children.push(...nodes);} replaceChildren(){this.children=[];}
  setAttribute(name,value){this.attrs[name]=value;} addEventListener(){} querySelectorAll(){return [];}
  text(){return this.textContent+this.children.map(child=>child.text()).join(" ");}
}
const nodes={}; const el=id=>nodes[id]||(nodes[id]=new Element());
const document={createElement:()=>new Element()};
let ablationScheme=null; const messages=[];
function ablationMessage(message,kind){messages.push([message,kind]);}
function updateAblationActions(){}
__HELPERS__
const normalMetrics={t2:{exceedance_rate_95:.01,exceedance_rate_99:0},spe:{exceedance_rate_95:.02,exceedance_rate_99:0},
  overall:{exceedance_rate_95:.03,exceedance_rate_99:0},continuous_false_alarm_event_count_95:1,longest_continuous_false_alarm_minutes:5};
const abnormalMetrics={valid_window_count:3,t2_detected_window_count_95:3,t2_detected_window_count_99:2,
  spe_detected_window_count_95:2,spe_detected_window_count_99:1,detection_rate_95:1,detection_rate_99:.67,
  detection_coverage_rate_95:.9,continuous_detection_event_count_95:3};
const scheme=(name,dropped,common,match)=>({name,dropped_tag:dropped,status:'evaluated',tag_count:6,dynamic_features:10,
  training_rows:299,training_common_rows:common,training_timestamps_match:match,n_components:2,cumulative_explained_variance:.9,
  validation_errors:[],conclusion:'可考虑剔除',reasons:[],detection_delta:{detection_rate_95:0,t2_detected_window_count_95:-1},
  validation:{normal_validation:{original_rows:110,common_rows:110,coverage:1,common_coverage:1,metrics:normalMetrics},
    known_abnormal:{original_rows:120,common_rows:120,coverage:1,common_coverage:1,metrics:abnormalMetrics}}});
renderAblationSchemes({warnings:[],schemes:[scheme('基准模型',null,299,true),scheme('方案 1','A',250,false)]});
const text=el('ablationResults').text();
for(const expected of ['共同训练样本','6 / 10 / 299 / 250','T²检出窗口 95% / 99% 3 / 2','T²检出窗口95%：-1 个窗口',
  '不一致或未确认：训练样本时间戳不同，变化不能全部归因于变量剔除，存在可比性限制']) assert(text.includes(expected),expected+'|'+text);
assert(text.includes('6 / 10 / 299 / 299'),text);
assert(text.includes('训练时间戳一致'),text);
'''.replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_ablation_uses_one_workbench_and_existing_invalidation_entries():
    class Ids(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
            self.training_tabs = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            if "training-tool-tab" in attrs.get("class", "").split():
                self.training_tabs.append(attrs)

    parser = Ids()
    parser.feed(ui.INDEX_HTML)
    assert all(count == 1 for count in Counter(parser.ids).values())
    assert [item["id"] for item in parser.training_tabs] == ["trainingMainTab", "redundancyAblationTab"]
    assert all("candidate-tool-tab" not in item["class"] for item in parser.training_tabs)
    for entry, hook in (
        ("function invalidateQuality(reason)", "globalThis.invalidateRedundancyAblation?.(reason, true)"),
        ("function renderValidationWindows()", "globalThis.refreshRedundancyValidation?.()"),
        ("function renderValidation(data)", 'globalThis.invalidateRedundancyAblation?.("验证状态已更新")'),
        ("async function trainModel(modelPurpose)", 'globalThis.invalidateRedundancyAblation?.("正在重新训练", true)'),
    ):
        assert hook in ui.INDEX_HTML.split(entry, 1)[1].split("\n", 3)[1]


def test_ablation_real_browser_workflow(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ui._BASE_WEB, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(61)
    x = rng.normal(size=700)
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=700, freq="5min"),
                          "A": x, "B": x + rng.normal(0, .03, 700), "C": -x + rng.normal(0, .03, 700),
                          "D": rng.normal(size=700), "E": rng.normal(size=700), "F": rng.normal(size=700)})
    frame["label"] = "normal"
    for start, end in ((430, 469), (490, 529), (550, 589)):
        frame.loc[start:end, "D"] += 12
        frame.loc[start:end, "label"] = "known"
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    windows = [{"id": name, "type": kind, "start": frame.time.iloc[start].isoformat(),
                "end": frame.time.iloc[end].isoformat(), "enabled": True, "comment": "人工已知窗口"}
               for name, kind, start, end in [("normal", "normal_validation", 310, 419),
                                              ("bad1", "known_abnormal", 430, 469),
                                              ("bad2", "known_abnormal", 490, 529),
                                              ("bad3", "known_abnormal", 550, 589)]]
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],requests=[],reports=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(request.url().endsWith('/api/redundancy-ablation')) requests.push(request.postDataJSON());});
  try {
    const response=await page.goto(process.env.ABLATION_URL);
    assert.equal(require('node:crypto').createHash('sha256').update(await response.body()).digest('hex'),process.env.ABLATION_HTML_HASH);
    await page.evaluate(()=>document.fonts.ready);
    await page.waitForLoadState('networkidle');
    const repeatedVisibilityWrites=await page.evaluate(async()=>{
      let count=0;
      const observer=new MutationObserver(records=>{count+=records.length;});
      observer.observe(document.getElementById('releaseContent'),{attributes:true,attributeFilter:['hidden']});
      await new Promise(resolve=>setTimeout(resolve,120));
      observer.disconnect();return count;
    });
    assert.equal(repeatedVisibilityWrites,0,'unchanged workflow visibility must not retrigger its observer');
    await page.evaluate(()=>showWorkflowStage('modelPanel'));
    await page.locator('#redundancyAblationTab').click();
    assert(await page.locator('#ablationCandidatesButton').isDisabled());
    assert((await page.locator('#ablationStatus').innerText()).includes('请先完成'));
    await page.locator('#trainingMainTab').click();
    assert(await page.locator('#trainingMainPanel').isVisible());
    assert(!(await page.locator('#redundancyAblationPanel').isVisible()));
    async function action(id,endpoint='/api/redundancy-ablation') {
      const waiting=page.waitForResponse(r=>r.url().endsWith(endpoint));
      await page.locator('#'+id).click();
      const response=await waiting,data=await response.json();assert(response.ok(),JSON.stringify(data));
      return data;
    }
    await page.evaluate(()=>showWorkflowStage('configPanel'));
    await page.locator('#fileInput').setInputFiles(process.env.ABLATION_CSV);
    await action('uploadButton','/api/upload');
    await page.locator('#timestampColumn').selectOption('time');
    await action('inspectButton','/api/inspect');
    // Fixture preparation uses the existing confirmed-window state; subsequent training and Tab actions use real APIs.
    await page.evaluate(()=>{
      state.trainingWindows=[{id:'training-ui',start:'2026-01-01T00:00:00',end:'2026-01-02T00:55:00',source:'manual',source_ref:null,comment:'浏览器训练窗口',enabled:true}];
      el('maxLag').value='5';el('lagStep').value='5';el('filterMethod').value='none';el('components').value='2';
      invalidateQuality('准备训练');renderTrainingWindows();updateQualityButtonAvailability();showWorkflowStage('modelPanel');
    });
    await action('qualityButton','/api/quality');
    assert(!(await page.locator('#trainButton').isDisabled()),await page.locator('#qualityIssues').innerText());
    const trained=await action('trainButton','/api/train');
    await page.waitForFunction(()=>state.training?.model_status==='candidate');
    fs.mkdirSync(process.env.ABLATION_SCREENSHOTS,{recursive:true});
    await page.screenshot({path:path.join(process.env.ABLATION_SCREENSHOTS,'training-1440.png'),fullPage:true});
    await page.locator('#redundancyAblationTab').click();
    const candidates=await action('ablationCandidatesButton');
    assert(await page.locator('#ablationCandidatesButton').evaluate(e=>e.classList.contains('secondary')));
    assert(!await page.locator('#ablationEvaluateButton').evaluate(e=>e.classList.contains('secondary')));
    assert((await page.locator('#ablationGroup').innerText()).includes('候选组 1'));
    assert(candidates.groups.length>0);
    assert(!('model_sha256' in requests.at(-1)));
    assert((await page.locator('#ablationPolicy').innerText()).includes('100'));
    assert((await page.locator('#ablationPolicy').innerText()).includes('30'));
    await page.locator('#ablationGroup').selectOption(candidates.groups[0].id);
    assert.equal(await page.locator('#ablationEvidence tbody tr').count(),3);
    await page.getByRole('checkbox',{name:'锁定 A',exact:true}).check();
    assert(await page.getByRole('checkbox',{name:'待剔除 A',exact:true}).isDisabled());
    await page.getByRole('checkbox',{name:'待剔除 C',exact:true}).check();
    assert(await page.getByRole('checkbox',{name:'待剔除 A',exact:true}).isDisabled());
    const noValidation=await action('ablationEvaluateButton');
    assert(noValidation.schemes.every(s=>s.conclusion==='证据不足'));
    assert((await page.locator('#ablationResults').innerText()).includes('异常检出能力尚未验证'));
    assert.equal(requests.at(-1).model_sha256,candidates.model_sha256);
    assert.deepEqual(requests.at(-1).locked_tags,['A']);
    assert.equal(requests.at(-1).encoding,'utf-8-sig');
    assert(!('max_lag_minutes' in requests.at(-1)));
    // The validation-window editor's render entry invalidates without demanding re-training.
    await page.evaluate(windows=>{state.validationWindows=windows;renderValidationWindows();},JSON.parse(process.env.ABLATION_WINDOWS));
    assert.equal(await page.locator('#ablationResults tbody tr').count(),0);
    await page.evaluate(()=>showWorkflowStage('validationPanel'));
    await page.locator('#labelColumn').selectOption('label');
    await page.evaluate(()=>showWorkflowStage('modelPanel'));
    await action('ablationCandidatesButton');
    await page.locator('#ablationGroup').selectOption(candidates.groups[0].id);
    await page.getByRole('checkbox',{name:'锁定 A',exact:true}).check();
    await page.getByRole('checkbox',{name:'待剔除 C',exact:true}).check();
    const evaluated=await action('ablationEvaluateButton');
    assert.equal(requests.at(-1).label_column,'label');
    assert.equal(evaluated.schemes[1].validation.normal_validation.common_rows,110);
    assert.equal(evaluated.schemes[1].validation.known_abnormal.common_rows,120);
    assert((await page.locator('#ablationResults').innerText()).includes('工程标签'));
    assert.equal(await page.locator('#ablationResults tbody tr').count(),2);
    const trainingCell=await page.locator('#ablationResults tbody tr').nth(1).locator('td').nth(7).innerText();
    const dropped=evaluated.schemes[1];
    assert.equal(trainingCell,`${dropped.tag_count} / ${dropped.dynamic_features} / ${dropped.training_rows} / ${dropped.training_common_rows}`);
    assert.equal(evaluated.schemes[1].training_common_rows,299);
    assert.deepEqual(await page.locator('#ablationResults thead th').allTextContents(),['选择','方案 / 剔除Tag','结论 / 原因','状态','正常验证','已知异常验证','检出变化','Tag数 / 动态维度 / 训练样本 / 共同训练样本','主元 / 累计解释率','训练时间戳一致']);
    // Save stable artifacts for human visual review, and check only declared safety boundaries.
    fs.mkdirSync(process.env.ABLATION_SCREENSHOTS,{recursive:true});
    for(const width of [1440,900,390]) {
      await page.setViewportSize({width,height:1000});
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const report=await page.locator('#redundancyAblationPanel').evaluate(panel=>({width:innerWidth,
        overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
        right:panel.getBoundingClientRect().right,display:getComputedStyle(panel).display,
        actions:[...panel.querySelectorAll('.actions button')].map(b=>({right:b.getBoundingClientRect().right,width:b.getBoundingClientRect().width})),
        tables:[...panel.querySelectorAll(':scope > .table-wrap')].map(t=>({right:t.getBoundingClientRect().right,scroll:t.scrollWidth>t.clientWidth}))}));
      assert.equal(report.overflow,0,JSON.stringify(report));assert.equal(report.display,'grid');
      assert(report.right<=width+1&&report.actions.every(b=>b.right<=width+1&&b.width>0),JSON.stringify(report));
      assert(report.tables.every(t=>t.right<=width+1),JSON.stringify(report));
      if(width===390) assert(report.tables.some(t=>t.scroll));
      reports.push(report);
      if(width!==900) await page.screenshot({path:path.join(process.env.ABLATION_SCREENSHOTS,`ablation-${width}.png`),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.getByRole('radio',{name:'选择方案 '+evaluated.schemes[1].name,exact:true}).check();
    // Existing model, download and validation must survive confirmed application.
    const before=await page.evaluate(()=>{state.validation={marker:'existing-validation'};return JSON.stringify([state.training,state.validation,state.trainingWindows,el('modelDownload').href]);});
    page.once('dialog',dialog=>dialog.dismiss());await page.locator('#ablationApplyButton').click();
    assert(await page.evaluate(()=>state.selectedModelTags.has('C')));
    page.once('dialog',dialog=>dialog.accept());await page.locator('#ablationApplyButton').click();
    assert(!(await page.evaluate(()=>state.selectedModelTags.has('C'))));
    assert.equal(await page.evaluate(()=>JSON.stringify([state.training,state.validation,state.trainingWindows,el('modelDownload').href])),before);
    assert(await page.locator('#trainButton').isDisabled());assert(await page.locator('#trainingMainPanel').isVisible());
    assert(await page.locator('#ablationApplyButton').isDisabled());
    // Controlled response cases exercise failures and late arrivals against the actual assembled page.
    async function restore() {
      await page.evaluate(data=>{state.selectedModelTags.add('C');state.validation=null;state.runId=data.run_id;state.training=data;renderTraining(data);showWorkflowStage('modelPanel');},trained);
      await page.locator('#redundancyAblationTab').click();
    }
    await restore();
    const fixture=structuredClone(candidates);
    fixture.groups[0].pairs[2].high_correlation=false;fixture.groups[0].pairs[2].pearson_r=.91;
    fixture.diagnostics.unavailable_pairs=[{tag_a:'A',tag_b:'constant',valid_pair_count:299,unavailable_reason:'配对样本中存在常量变量'}];
    let reply=fixture;
    await page.route('**/api/redundancy-ablation',route=>route.fulfill({json:reply}));
    await action('ablationCandidatesButton');await page.locator('#ablationGroup').selectOption(fixture.groups[0].id);
    assert((await page.locator('#ablationEvidence').innerText()).includes('低于阈值'));
    await page.locator('#redundancyAblationPanel details summary').click();
    assert((await page.locator('#ablationDiagnostics').innerText()).includes('常量'));
    await page.getByRole('checkbox',{name:'待剔除 C',exact:true}).check();
    reply=structuredClone(evaluated);
    reply.schemes[1].validation_errors=[{window_id:'bad1',type:'known_abnormal',error:'测试验证失败'}];
    reply.schemes[1].validation.normal_validation.common_rows=0;reply.schemes[1].conclusion='证据不足';
    reply.schemes[1].training_timestamps_match=false;reply.schemes[1].training_common_rows=0;
    reply.schemes.push({name:'不可训练方案',dropped_tag:'B',status:'failed',error:'固定主元数不合法',validation_errors:[]});
    await action('ablationEvaluateButton');
    assert((await page.locator('#ablationStatus').innerText()).includes('验证不完整'));
    const resultText=await page.locator('#ablationResults').innerText();
    for(const text of ['验证不完整','bad1（known_abnormal）：测试验证失败','无共同有效验证样本','固定主元数不合法','训练样本时间戳不同，变化不能全部归因于变量剔除','T²检出窗口 95% / 99%']) assert(resultText.includes(text),resultText);
    assert(resultText.includes(`${reply.schemes[1].tag_count} / ${reply.schemes[1].dynamic_features} / ${reply.schemes[1].training_rows} / 0`),resultText);
    assert(await page.getByRole('radio',{name:'选择方案 不可训练方案',exact:true}).isDisabled());
    // Hash mismatch may not replace the displayed results or become applicable.
    reply={...evaluated,model_sha256:'wrong-hash'};await action('ablationEvaluateButton');
    await page.waitForFunction(()=>el('ablationStatus').textContent.includes('身份不匹配'));
    assert.equal(await page.locator('#ablationResults tbody tr').count(),0);
    await page.unroute('**/api/redundancy-ablation');await restore();
    // Out-of-order response after a configuration change is ignored, even if run ID is unchanged.
    let delayed,resolveArrival;
    const arrival=new Promise(resolve=>resolveArrival=resolve);
    await page.route('**/api/redundancy-ablation',async route=>{delayed=route;resolveArrival();});
    await page.locator('#ablationCandidatesButton').click();await arrival;
    await page.evaluate(()=>invalidateQuality('建模配置已修改'));
    await delayed.fulfill({json:fixture});
    await page.waitForLoadState('networkidle');
    assert((await page.locator('#ablationStatus').innerText()).includes('已失效'));
    assert.equal(await page.locator('#ablationGroup option').count(),1);
    assert(await page.locator('#ablationCandidatesButton').isDisabled());
    await page.unroute('**/api/redundancy-ablation');await restore();
    // Unsupported lifecycle and keyboard navigation do not interfere with candidate tabs.
    await page.evaluate(()=>{state.training={...state.training,model_purpose:'exploratory',model_status:'draft'};});
    await page.locator('#redundancyAblationTab').click();assert(await page.locator('#ablationCandidatesButton').isDisabled());
    await page.locator('#redundancyAblationTab').press('ArrowLeft');assert(await page.locator('#trainingMainPanel').isVisible());
    assert.equal(await page.locator('.candidate-tool-tab').count(),5);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({run_id:trained.run_id,model_sha256:candidates.model_sha256,reports,requests:requests.length}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
"""
    screenshot_dir = Path(os.environ.get("ABLATION_SCREENSHOTS", str(tmp_path / "screenshots")))
    env = {**os.environ, "ABLATION_URL": f"http://127.0.0.1:{server.server_port}",
           "ABLATION_CSV": str(csv), "ABLATION_WINDOWS": json.dumps(windows),
           "ABLATION_HTML_HASH": hashlib.sha256(ui.INDEX_HTML.encode()).hexdigest(),
           "ABLATION_SCREENSHOTS": str(screenshot_dir)}
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], env=env, capture_output=True, text=True,
                                encoding="utf-8", timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.strip().splitlines()[-1])
        model = tmp_path / "runs" / report["run_id"] / "model.pcamodel"
        assert hashlib.sha256(model.read_bytes()).hexdigest() == report["model_sha256"]
        assert len(report["reports"]) == 3
        print(json.dumps(report, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
