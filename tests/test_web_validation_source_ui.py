"""Execute the actual validation-source JavaScript against deferred API responses."""

import pytest

from pca_model_builder import web
from test_web import _run_node_javascript


def _run_source_ui(scenario):
    html = web.INDEX_HTML
    handlers = html[html.index("function currentValidationSource()"):
                    html.index('el("recordValidationDecision").addEventListener')]
    invalidation = html[html.index("function invalidateValidationInvestigation()"):
                        html.index("function currentValidationInvestigationContext(")]
    harness = r'''
const assert=require('node:assert/strict'),nodes={},pending=[],messages=[],renders=[];
class Element {
  constructor(){this.value='';this.textContent='';this.hidden=false;this.disabled=false;
    this.href='#';this.dataset={};this.children=[];this.listeners={};this.option={};
    this.files=[];this.classList={add(){},remove(){}};}
  addEventListener(event,handler){this.listeners[event]=handler;}
  append(...items){this.children.push(...items);} replaceChildren(){this.children=[];}
  querySelector(){return this.option;} get rows(){return this.children;}
  setAttribute(){} removeAttribute(){} focus(){}
}
const el=id=>nodes[id]??=new Element();
const document={createElement:()=>new Element(),querySelectorAll:selector=>
  selector==='#validationWindowTable tr'?el('validationWindowTable').children:[]};
class FormData {constructor(){this.entries=[];} append(...entry){this.entries.push(entry);}}
function fillSelect(node,values){node.options=[...values];node.value=values[0]||'';}
function displayTime(value){return value;}
function displayUiValue(value){return value;}
function candidateId(){return 'new';}
function setBusy(button,busy){button.disabled=busy;}
function setStatus(message,level){messages.push({message,level});}
function renderValidation(data){renders.push(data);el('validationContent').hidden=false;}
function api(url,options){return new Promise((resolve,reject)=>pending.push({url,options,resolve,reject}));}
const inspection={columns:['time','A','train_label'],time_start:'2026-01-01T00:00',
  suggested_validation_start:'2026-01-02T00:00',time_end:'2026-01-02T23:55'};
const independent={columns:['vtime','A','工程标签'],time_start:'2026-02-01T00:00',
  time_end:'2026-02-02T23:55',rows:100,quality_issues:[]};
const state={fileId:'training-file',fileName:'training.csv',inspection,
  training:{model_purpose:'normal_state',training_window_summary:[]},runId:'run-original',
  tags:['A'],windows:[{id:'training-window'}],validationWindows:[],validation:null};
el('timestampColumn').value='time';el('encoding').value='utf-8-sig';
el('validationSource').value='current';el('validationEncoding').value='utf-8';
const trainingRefs=Object.fromEntries(['fileId','inspection','training','runId','tags','windows'].map(k=>[k,state[k]]));
const trainingJson=JSON.stringify(trainingRefs);
function assertTraining(){for(const [key,value] of Object.entries(trainingRefs))assert.equal(state[key],value,key);
  assert.equal(JSON.stringify(trainingRefs),trainingJson);}
function fire(id,event='click'){return el(id).listeners[event]();}
function switchSource(value){el('validationSource').value=value;fire('validationSource','change');}
function readyUpload(id='validation-file'){
  state.validationUpload={file_id:id,filename:id+'.csv',timestamp_column:'vtime',encoding:'utf-8',inspection:independent};
  el('validationTimestampColumn').value='vtime';return state.validationUpload;
}
function assertCleared(){assert.deepEqual(state.validationWindows,[]);assert.equal(state.validation,null);
  assert.equal(el('validationDecision').value,'hold');assert.equal(el('validationDecisionComment').value,'');
  assert.equal(el('recordValidationDecision').disabled,true);assert.equal(el('freezeDeployment').disabled,true);
  assert.equal(el('validationContent').hidden,true);assert.equal(el('validationEmpty').hidden,false);
  for(const id of ['validatedModelDownload','frozenModelDownload','deploymentModelDownload']){
    assert.equal(el(id).hidden,true);assert.equal(el(id).href,'#');}
}
__INVALIDATION__
__HANDLERS__
(async()=>{
__SCENARIO__
assertTraining();assert.equal(pending.length,0);
console.log(JSON.stringify({verified:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    script = (harness.replace('__INVALIDATION__', invalidation)
              .replace('__HANDLERS__', handlers).replace('__SCENARIO__', scenario))
    assert _run_node_javascript(script)['verified']


def test_independent_upload_inspect_and_source_reset_preserve_training():
    _run_source_ui(r'''
switchSource('upload');assert.equal(currentValidationSource(),null);assert.equal(el('validateButton').disabled,true);
await fire('validationUploadButton');assert.equal(pending.length,0);assert.match(messages.at(-1).message,/请选择/);
await fire('validationInspectButton');assert.equal(pending.length,0);assert.match(messages.at(-1).message,/先上传/);
el('validationFileInput').files=[{name:'independent.csv'}];fire('validationFileInput','change');
const upload=fire('validationUploadButton'),request=pending.shift();assert.equal(request.url,'/api/upload');
assert.deepEqual(request.options.body.entries,[['file',el('validationFileInput').files[0]]]);
request.resolve({file_id:'independent',filename:'independent.csv',columns:independent.columns,encoding:'utf-8'});
await upload;assertTraining();assert.equal(state.validationUpload.file_id,'independent');
assert.equal(el('validationInspectButton').disabled,false);assert.equal(el('validateButton').disabled,true);
const inspect=fire('validationInspectButton'),check=pending.shift();assert.equal(check.url,'/api/inspect');
assert.deepEqual(JSON.parse(check.options.body),{file_id:'independent',timestamp_column:'vtime',encoding:'utf-8'});
check.resolve({...independent,quality_issues:[{code:'missing_values',count:3}]});await inspect;
assert.equal(currentValidationSource(),state.validationUpload);assert.equal(el('validateButton').disabled,false);
assert.equal(el('validationStart').value,independent.time_start);assert.deepEqual(el('labelColumn').options,independent.columns);
assert.match(el('validationSourceStatus').textContent,/missing_values（3）/);assertTraining();
for(const mode of ['current','upload']){
  state.validationWindows=[{id:'old'}];state.validation={old:true};
  el('validationDecision').value='passed';el('validationDecisionComment').value='old decision';
  for(const id of ['validatedModelDownload','frozenModelDownload','deploymentModelDownload']){el(id).hidden=false;el(id).href='old';}
  switchSource(mode);assertCleared();assert.equal(el('validationUploadControls').hidden,mode==='current');
  assert.equal(el('validationStart').value,mode==='current'?inspection.suggested_validation_start:independent.time_start);
}
for(const [id,event] of [['validationTimestampColumn','change'],['validationEncoding','input'],['validationFileInput','change']]){
  readyUpload();fire(id,event);assert.equal(currentValidationSource(),null);assert.equal(el('validateButton').disabled,true);assertCleared();
}
''')


@pytest.mark.parametrize('failure', ['invalid_timestamp', 'unsorted_timestamp', 'duplicate_timestamp', 'parse'])
def test_inspection_errors_keep_validation_disabled(failure):
    _run_source_ui(r'''
switchSource('upload');readyUpload();
const task=fire('validationInspectButton'),request=pending.shift();
if('__FAILURE__'==='parse')request.reject(new Error('encoding/parse failed'));
else request.resolve({...independent,quality_issues:[{code:'__FAILURE__'}]});
await task;assert.equal(state.validationUpload.inspection,null);assert.equal(currentValidationSource(),null);
assert.equal(el('validateButton').disabled,true);assert.match(el('validationSourceStatus').textContent,/检查失败/);
assert.match(el('validationSourceStatus').textContent,/__EXPECTED__/);assertCleared();
'''.replace('__FAILURE__', failure).replace('__EXPECTED__', 'parse failed' if failure == 'parse' else '时间戳'))


@pytest.mark.parametrize('mode', ['current', 'upload'])
def test_replay_payload_uses_selected_source_and_engineering_label(mode):
    _run_source_ui(r'''
readyUpload();switchSource('__MODE__');el('labelColumn').value='__LABEL__';
const task=fire('validateButton'),request=pending.shift();assert.equal(request.url,'/api/validate');
assert.deepEqual(JSON.parse(request.options.body),{
  run_id:'run-original',file_id:'__FILE__',timestamp_column:'__TIME__',encoding:'__ENCODING__',
  validation_windows:[{id:'validation-default-001',type:'normal_validation',start:el('validationStart').value,
    end:el('validationEnd').value,enabled:true,comment:''}],label_column:'__LABEL__'});
const report={source:'__MODE__'};request.resolve(report);await task;
assert.equal(state.validation,report);assert.deepEqual(renders,[report]);assert.equal(el('validationDecision').value,'hold');
'''.replace('__MODE__', mode).replace('__LABEL__', '工程标签' if mode == 'upload' else 'train_label')
        .replace('__FILE__', 'validation-file' if mode == 'upload' else 'training-file')
        .replace('__TIME__', 'vtime' if mode == 'upload' else 'time')
        .replace('__ENCODING__', 'utf-8' if mode == 'upload' else 'utf-8-sig'))


@pytest.mark.parametrize('operation', ['upload', 'inspect'])
@pytest.mark.parametrize('failure', [False, True], ids=['success', 'failure'])
@pytest.mark.parametrize('old_first', [False, True], ids=['old-last', 'old-first'])
def test_stale_requests_cannot_replace_new_source(operation, failure, old_first):
    _run_source_ui(r'''
switchSource('upload');readyUpload('old-file');el('validationFileInput').files=[{name:'old.csv'}];
const button={upload:'validationUploadButton',inspect:'validationInspectButton'}['__OP__'];
const oldTask=fire(button),old=pending.shift();
// Switch away and back to catch ABA source changes, then start a newer request.
switchSource('current');switchSource('upload');const fresh=readyUpload('new-file');
const newTask=fire(button),latest=pending.shift();
function finishOld(){if(__FAILURE__)old.reject(new Error('obsolete failure'));
  else old.resolve('__OP__'==='upload'?{file_id:'obsolete',filename:'obsolete.csv',columns:['obsolete']}:
    {...independent,columns:['obsolete']});}
if(__OLD_FIRST__){finishOld();await oldTask;}
const response='__OP__'==='upload'?{file_id:'newest',filename:'newest.csv',columns:independent.columns,encoding:'utf-8'}:
  {...independent,rows:999};
latest.resolve(response);await newTask;
const source=state.validationUpload,report=state.validation,status=el('validationSourceStatus').textContent;
if(!__OLD_FIRST__){finishOld();await oldTask;}
assert.equal(state.validationUpload,source);assert.equal(state.validation,report);
assert.equal(el('validationSourceStatus').textContent,status);
assert(!messages.some(item=>item.message.includes('obsolete failure')));
if('__OP__'==='upload'){assert.equal(source.file_id,'newest');assert.equal(el('validationInspectButton').disabled,false);}
if('__OP__'==='inspect'){assert.equal(source,fresh);assert.equal(source.inspection,response);assert.equal(el('validateButton').disabled,false);}
'''.replace('__OP__', operation).replace('__FAILURE__', str(failure).lower())
        .replace('__OLD_FIRST__', str(old_first).lower()))


@pytest.mark.parametrize('failure', [False, True], ids=['success', 'failure'])
@pytest.mark.parametrize('mode', ['current', 'upload'])
def test_pending_replay_blocks_parallel_requests_until_stale_completion(failure, mode):
    _run_source_ui(r'''
readyUpload('old-file');switchSource('upload');
const oldTask=fire('validateButton'),old=pending.shift();
assert.equal(state.validationReplayPending,true);assert.equal(el('validateButton').disabled,true);
await fire('validateButton');assert.equal(pending.length,0);
switchSource('current');switchSource('upload');readyUpload('new-file');switchSource('__MODE__');
assertCleared();assert.equal(el('validateButton').disabled,true);
await fire('validateButton');assert.equal(pending.length,0);assert.equal(state.validationReplayPending,true);
assert.equal(el('validationSource').disabled,false);assert.equal(el('validationFileInput').disabled,false);
if(__FAILURE__)old.reject(new Error('obsolete replay failure'));else old.resolve({obsolete:true});
await oldTask;assert.equal(state.validationReplayPending,false);assert.equal(el('validateButton').disabled,false);
assert.equal(state.validation,null);assert.deepEqual(renders,[]);
assert(!messages.some(item=>item.message.includes('obsolete replay failure')));
const task=fire('validateButton'),request=pending.shift();assert.equal(request.url,'/api/validate');
assert.equal(JSON.parse(request.options.body).file_id,'__FILE__');
const report={newest:true};request.resolve(report);await task;
assert.equal(state.validation,report);assert.deepEqual(renders,[report]);assert.equal(state.validationReplayPending,false);
'''.replace('__FAILURE__', str(failure).lower()).replace('__MODE__', mode)
        .replace('__FILE__', 'training-file' if mode == 'current' else 'new-file'))


def test_upload_failure_and_replay_parse_failure_allow_retry():
    _run_source_ui(r'''
switchSource('upload');el('validationFileInput').files=[{name:'broken.csv'}];
const upload=fire('validationUploadButton');pending.shift().reject(new Error('unsupported file'));
await upload;assert.equal(state.validationUpload,null);assert.equal(el('validationUploadButton').disabled,false);
assert.equal(el('validationInspectButton').disabled,true);assert.equal(el('validateButton').disabled,true);
assert.match(el('validationSourceStatus').textContent,/上传失败.*unsupported file/);
readyUpload();switchSource('upload');
const replay=fire('validateButton');pending.shift().reject(new Error('validation parse failed'));
await replay;assert.equal(state.validation,null);assert.deepEqual(renders,[]);
assert.equal(state.validationReplayPending,false);assert.equal(el('validateButton').disabled,false);
assert.deepEqual(messages.at(-1),{message:'validation parse failed',level:'error'});
''')
