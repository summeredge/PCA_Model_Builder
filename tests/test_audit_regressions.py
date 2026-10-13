import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_model_results
from pca_model_builder.preprocessing import PreprocessingConfig, StateFilter, preprocess_window
from pca_model_builder.state_exploration import _preprocessing_summary


@pytest.mark.parametrize('filtered', [False, True])
def test_invalid_state_warmup_and_lag_losses_close(filtered):
    frame = pd.DataFrame({'A': np.sin(np.arange(100)), 'B': np.cos(np.arange(100)),
                          'STATE': [0] * 80 + [1] * 20},
                         index=pd.date_range('2026-01-01', periods=100, freq='5min'))
    frame.loc[frame.index[:60], 'A'] = np.nan
    config = PreprocessingConfig(5, 10 if filtered else 0, 10 if filtered else 0, 5,
        filter_method='trailing_mean' if filtered else 'none',
        state_filters=(StateFilter('STATE', minimum=1),) if filtered else ())
    processed = preprocess_window(frame, ['A', 'B'], config, include_intermediates=True)
    summary = _preprocessing_summary(processed)
    assert summary['loss_counts']['input_invalid_loss'] == 60
    assert summary['state_filter_loss'] == (20 if filtered else 0)
    assert summary['loss_count'] + processed.summary.final_dynamic_row_count == 100


def test_training_response_rejects_quality_changes():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    html = web_model_results.INDEX_HTML
    train = 'async function trainModel' + html.split('async function trainModel', 1)[1].split('\nel("trainExploratoryButton")', 1)[0]
    invalidate = 'function invalidateQuality' + html.split('function invalidateQuality', 1)[1].split('\nfunction firstOrderAlphaError', 1)[0]
    script = r'''
const assert=require('node:assert/strict');
const nodes={};function el(id){return nodes[id]??={value:'',disabled:true,hidden:false};}
const document={querySelector:()=>({click(){}})};
const state={qualityRevision:0,registry:{},excludedTags:[],trainingWindows:[],candidateWindows:[]};
let eligibilityRevision=0,resolve,reject;
const statuses=[],rendered=[];
function api(){return new Promise((r,j)=>{resolve=r;reject=j;});}
function selectedTags(){return ['A','B'];} function commonPayload(){return {};}
function trainingWindowsPayload(){return state.trainingWindows;} function numberValue(){return 0;}
function performanceConfigPayload(){return null;} function setStatus(message,type='info'){statuses.push({message,type});} function renderTraining(data){rendered.push(data);}
function setBusy(b,v){b.disabled=v;b.busy=v;} function invalidateValidationInvestigation(){}
function renderModelTrainingDataSummary(){} function renderModelQualityStatus(){} function renderCurrentTagQuality(){}
''' + train + '\n' + invalidate + r'''
(async()=>{
for(const change of ['Lag 10→15','Tag','训练窗口','主元数','滤波','不变']){
 state.quality={can_train:true};state.qualityStatus='passed';el('trainButton').disabled=false;
 state.training=null;state.runId=null;el('validateButton').disabled=true;const pending=trainModel('normal_state');
 if(change!=='不变')invalidateQuality(change);
 resolve({run_id:'new',model_purpose:'normal_state',training_rows:40,dynamic_features:2});await pending;
 assert.equal(state.runId,change==='不变'?'new':null);
 assert.equal(el('validateButton').disabled,change!=='不变');
 assert.equal(el('trainButton').disabled,change!=='不变');
 assert.equal(state.qualityStatus,change==='不变'?'passed':'changed');
}
for(const purpose of ['normal_state','exploratory']){
 const button=el(purpose==='exploratory'?'trainExploratoryButton':'trainButton');
 for(const allowed of [true,false,null]){
  state.quality={can_train:true};state.training=null;state.runId=null;
  const old=trainModel(purpose),resolveOld=resolve;
  invalidateQuality('配置 B');
  state.quality=allowed===null?null:{can_train:!allowed,training_readiness:{[purpose]:{can_train:allowed}}};
  state.qualityStatus=allowed===null?'checking':allowed?'passed':'issues';button.disabled=!allowed;
  resolveOld({run_id:'old',model_purpose:purpose});await old;
  assert.equal(state.runId,null);assert.equal(state.training,null);
  assert.equal(button.disabled,!allowed);assert.equal(button.busy,false);
 }
 for(const newerFinishesFirst of [false,true]){
  state.quality={can_train:true};state.runId=null;state.training=null;
  const old=trainModel(purpose),resolveOld=resolve;
  invalidateQuality('配置 B');
  state.quality={training_readiness:{[purpose]:{can_train:true}}};state.qualityStatus='passed';button.disabled=false;
  const newer=trainModel(purpose),resolveNewer=resolve;
  const currentRequest=button.trainingRequest;
  if(newerFinishesFirst){resolveNewer({run_id:'current',model_purpose:purpose});await newer;}
  resolveOld({run_id:'old',model_purpose:purpose});await old;
  assert.equal(button.busy,!newerFinishesFirst);assert.equal(button.disabled,!newerFinishesFirst);
  assert.equal(state.runId,newerFinishesFirst?'current':null);
  if(!newerFinishesFirst){
   assert.equal(button.trainingRequest,currentRequest);
   resolveNewer({run_id:'current',model_purpose:purpose});await newer;
  }
  assert.equal(state.runId,'current');assert.equal(button.disabled,false);assert.equal(button.busy,false);
 }
}
function trainingResult(runId,purpose){return {run_id:runId,model_purpose:purpose,training_rows:runId==='A'?11:22,dynamic_features:2};}
function resetTrainingResult(){
 state.training={run_id:'prior'};state.runId='prior';state.exploratoryRunId='prior-exploration';
 state.validationInvestigationModel={training:state.training,tags:['prior']};state.validation={run_id:'prior-validation'};state.trainingRequest=null;
 el('validateButton').disabled=false;
 for(const id of ['trainButton','trainExploratoryButton']){el(id).disabled=false;el(id).busy=false;delete el(id).trainingRequest;}
 statuses.length=0;rendered.length=0;
}
function assertTrainingResult(runId,purpose){
 assert.equal(state.runId,runId);
 assert.equal(state.exploratoryRunId,purpose==='exploratory'?runId:'prior-exploration');
 assert.equal(state.training.run_id,runId);assert.equal(state.validationInvestigationModel.training,state.training);
 assert.deepEqual(state.validationInvestigationModel.tags,['A','B']);assert.equal(state.validation,null);
 assert.equal(el('validateButton').disabled,purpose==='exploratory');
 assert.deepEqual(rendered.map(data=>data.run_id),[runId]);
 const success=statuses.filter(item=>item.type==='success');assert.equal(success.length,1);
 assert.match(success[0].message,runId==='B'?/22 个动态样本/:/11 个动态样本/);
}
for(const purpose of ['normal_state','exploratory']){
 const button=el(purpose==='exploratory'?'trainExploratoryButton':'trainButton');
 for(const newerFinishesFirst of [false,true]){
  resetTrainingResult();state.quality={training_readiness:{normal_state:{can_train:true},exploratory:{can_train:true}}};state.qualityStatus='passed';
  const beforeEligibility=eligibilityRevision,beforeQuality=state.qualityRevision;
  const older=trainModel(purpose),resolveOlder=resolve,olderRequest=state.trainingRequest;
  const newer=trainModel(purpose),resolveNewer=resolve,newerRequest=state.trainingRequest;
  assert.notEqual(olderRequest,newerRequest);assert.equal(eligibilityRevision,beforeEligibility);assert.equal(state.qualityRevision,beforeQuality);
  if(newerFinishesFirst){resolveNewer(trainingResult('B',purpose));await newer;assertTrainingResult('B',purpose);}
  resolveOlder(trainingResult('A',purpose));await older;
  if(newerFinishesFirst){
   assertTrainingResult('B',purpose);assert.equal(button.busy,false);assert.equal(button.disabled,false);
  }else{
   assert.equal(state.runId,'prior');assert.equal(state.exploratoryRunId,'prior-exploration');
   assert.equal(state.training.run_id,'prior');assert.equal(state.validationInvestigationModel.training.run_id,'prior');
   assert.deepEqual(state.validation,{run_id:'prior-validation'});assert.deepEqual(rendered,[]);
   assert.equal(statuses.filter(item=>item.type==='success').length,0);assert.equal(el('validateButton').disabled,false);
   assert.equal(button.busy,true);assert.equal(button.disabled,true);assert.equal(button.trainingRequest,newerRequest);
   resolveNewer(trainingResult('B',purpose));await newer;assertTrainingResult('B',purpose);
   assert.equal(button.busy,false);assert.equal(button.disabled,false);
  }
 }
}
for(const olderFails of [false,true]){
 resetTrainingResult();state.quality={training_readiness:{normal_state:{can_train:true},exploratory:{can_train:true}}};state.qualityStatus='passed';
 const older=trainModel('normal_state'),resolveOlder=resolve,rejectOlder=reject;
 const newer=trainModel('exploratory'),resolveNewer=resolve;
 resolveNewer(trainingResult('B','exploratory'));await newer;assertTrainingResult('B','exploratory');
 if(olderFails)rejectOlder(new Error('过期训练错误'));else resolveOlder(trainingResult('A','normal_state'));
 await older;assertTrainingResult('B','exploratory');
 assert.equal(el('trainButton').busy,false);assert.equal(el('trainButton').disabled,false);
 assert.equal(el('trainExploratoryButton').busy,false);assert.equal(el('trainExploratoryButton').disabled,false);
 assert.equal(statuses.filter(item=>item.type==='error').length,0);
}
})().catch(e=>{console.error(e);process.exit(1);});
'''
    result = subprocess.run([node, '-'], input=script, text=True, encoding='utf-8', capture_output=True)
    assert result.returncode == 0, result.stderr


def test_real_plotly_group_click_and_episode_resize():
    if not (shutil.which('node') and os.environ.get('WEB_GEOMETRY_PLAYWRIGHT') and os.environ.get('WEB_GEOMETRY_BROWSER')):
        pytest.skip('Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER')
    server = ThreadingHTTPServer(('127.0.0.1', 0), web_model_results.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
try {
const page=await browser.newPage({viewport:{width:1440,height:1000}});
await page.goto(process.env.AUDIT_TEST_URL);await page.waitForFunction(()=>!!globalThis.Plotly);
await page.evaluate(()=>{
document.querySelectorAll('.panel,.candidate-tool-panel,.inner-panel').forEach(p=>p.classList.add('active'));
el('explorationContent').hidden=false;
const rows=[1,1,2,2,3].map((g,i)=>({cluster_id:String(g),pc1:i,pc2:i%2,timestamp:`2026-01-01T00:${String(i*5).padStart(2,'0')}:00`,segment_id:0}));
state.exploration={exploration_run_id:'test',cluster_series:rows,cluster_series_full:rows,cluster_candidates:[],cluster_centers:{1:[.5,.5],2:[2.5,.5],3:[4,0]},cluster_quality:{group_profiles:[{cluster_id:'1',episodes:[{start:rows[0].timestamp,end:rows[1].timestamp}]},{cluster_id:'2',episodes:[{start:rows[2].timestamp,end:rows[3].timestamp}]},{cluster_id:'3',episodes:[{start:rows[4].timestamp,end:rows[4].timestamp}]}]}};
state.explorationColorValues=new Map([['A',[10,null,10,20,null]]]);
el('explorationPcColorMode').add(new Option('A','variable:A'));el('explorationPcColorMode').value='variable:A';
renderExplorationPcChart(state.exploration);
state.groupProfileSelection={state_exploration:'2'};applyGroupProfileFocus('state_exploration',true);
});
await page.waitForFunction(()=>state.preferredRegionPlot?.data.length===7);
const snapshot=await page.evaluate(()=>JSON.stringify(state.exploration));
assert.equal(await page.locator('#explorationPcChart .legendtoggle').count(),3);
for(const [click,index] of [0,1,2,0,1,2].entries()) {
 await page.locator('#explorationPcChart .legendtoggle').nth(index).click();
 await page.waitForFunction(({index,click})=>state.preferredRegionPlot.data.filter(t=>t.legendgroup===`cluster:${index+1}`).every(t=>click<3?t.visible==='legendonly':t.visible!=='legendonly'),{index,click});
 const group=String(index+1);
 const traces=await page.evaluate(g=>state.preferredRegionPlot.data.filter(t=>t.legendgroup===`cluster:${g}`).map(t=>t.visible),group);
 // First three clicks hide every group; the following three restore them.
 const clickIndex=await page.evaluate(()=>globalThis.auditClicks=(globalThis.auditClicks||0)+1);
 assert(traces.every(v=>clickIndex<=3?v==='legendonly':v!=='legendonly'),JSON.stringify({clickIndex,group,traces}));
 assert.equal(await page.evaluate(()=>JSON.stringify(state.exploration)),snapshot);
}
for(const width of [1440,900,390,1440]) {
 await page.setViewportSize({width,height:1000});await page.waitForTimeout(150);
 const delta=await page.evaluate(()=>{
 const svg=el('explorationTimeline').querySelector('svg'),focus=svg.querySelector('[data-profile-episode]');
 const w=svg.viewBox.baseVal.width;return Math.max(Math.abs(Number(focus.getAttribute('x'))-(94+.4*(w-112))),Math.abs(Number(focus.getAttribute('width'))-.4*(w-112)));
 });assert(delta<1);
}
await page.evaluate(()=>{el('explorationPcColorMode').value='cluster';renderExplorationPcChart(state.exploration);});
await page.waitForFunction(()=>state.preferredRegionPlot?.data.length===6);
await page.locator('#explorationPcChart .legendtoggle').first().click();await page.waitForFunction(()=>state.preferredRegionPlot.data.filter(t=>t.legendgroup==='cluster:1').every(t=>t.visible==='legendonly'));
assert(await page.evaluate(()=>state.preferredRegionPlot.data.filter(t=>t.legendgroup==='cluster:1').every(t=>t.visible==='legendonly')));
}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
'''
    try:
        env = dict(os.environ, AUDIT_TEST_URL=f'http://127.0.0.1:{server.server_port}')
        result = subprocess.run([shutil.which('node'), '-'], input=script, text=True, encoding='utf-8', capture_output=True, env=env, timeout=90)
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
