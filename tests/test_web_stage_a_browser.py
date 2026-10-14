"""Stage A controls and editing use the real workbench DOM."""
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from pca_model_builder import web_model_results as ui


def test_stage_a_controls_and_validation_in_chromium():
    if not all((shutil.which("node"), os.environ.get("WEB_GEOMETRY_PLAYWRIGHT"), os.environ.get("WEB_GEOMETRY_BROWSER"))):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 try {
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.STAGE_A_URL);
  await page.evaluate(()=>{
   const raw=['ready','usable','disabled','normal','attention','abnormal','review','blocking','pending','accepted','rejected','used','dropped'];
   window.statusCases=[...raw.map(key=>({value:key,key})),...raw.map(key=>({value:displayUiValue(key)})),{value:'禁用'}];
   const table=document.createElement('table');table.innerHTML='<thead><tr><th>Tag</th><th>状态</th><th>备注</th></tr></thead><tbody>'+statusCases.map(({value})=>`<tr><td>${value}</td><td>${value}</td><td>${value}</td></tr>`).join('')+'</tbody>';
   const wrap=document.createElement('div');wrap.className='table-wrap';wrap.id='statusRegression';wrap.append(table);document.body.append(wrap);
   state.validationWindows=[];renderValidationWindows();
   if(el('validationDecisionStatus').textContent.includes('失效')) throw Error('initial invalidation');
   state.validationWindows=[{id:'normal-original',type:'normal_validation',enabled:true,start:'2026-01-01T00:00',end:'2026-01-02T00:00',comment:'original'}];
   renderValidationWindows();showWorkflowStage('validationPanel');
  });
  await page.waitForFunction(()=>document.querySelectorAll('#statusRegression .status-label').length===statusCases.length);
  const badges=await page.evaluate(()=>{
   const palette={};for(const [tone,color] of Object.entries({normal:'var(--normal)',attention:'#8a5a00',abnormal:'var(--danger)',neutral:'var(--muted)'})){const ref=document.createElement('span');ref.style.color=color;document.body.append(ref);palette[tone]=getComputedStyle(ref).color;ref.remove();}
   return [...document.querySelectorAll('#statusRegression tbody tr')].map((row,index)=>{
    const badge=row.cells[1].firstElementChild;return {...statusCases[index],text:badge.textContent,tag:badge.tagName,tone:badge.dataset.tone,rawKey:badge.dataset.status,color:getComputedStyle(badge).color,expectedColor:palette[badge.dataset.tone],ordinary:[row.cells[0].textContent,row.cells[2].textContent],ordinaryBadges:row.cells[0].children.length+row.cells[2].children.length};
   });
  });
  for(const badge of badges){
   const tone=({'正常':'normal','可用':'normal','已接受':'normal','已使用':'normal','关注':'attention','需确认':'attention','待决策':'attention','异常':'abnormal','阻止':'abnormal','已拒绝':'abnormal','已丢弃':'abnormal','已禁用':'neutral'})[badge.text];
   assert.equal(badge.tag,'SPAN');assert.equal(badge.tone,tone);assert.equal(badge.color,badge.expectedColor);assert.equal(badge.rawKey,badge.key);
   assert.deepEqual(badge.ordinary,[badge.value,badge.value]);assert.equal(badge.ordinaryBadges,0);
  }
  assert.equal(badges.find(b=>b.key==='disabled').text,'已禁用');
  assert.equal(badges.find(b=>b.value==='禁用').text,'已禁用');
  assert.equal(await page.locator('#statusRegression .status-label.ready').count(),1);
  assert.equal(await page.locator('#statusRegression .status-label.usable').count(),1);
  await page.locator('#validationWindowTable button').first().click();
  assert.equal(await page.locator('#validationWindowTable tr.editing').count(),1);
  await page.locator('#validationComment').fill('unsaved');await page.locator('#cancelValidationEdit').click();
  assert.equal(await page.evaluate(()=>state.validationWindows[0].comment),'original');
  assert.equal(await page.locator('#validationWindowTable tr.editing').count(),0);
  await page.locator('#validationWindowTable button').first().click();
  await page.locator('#validationComment').fill('saved');await page.locator('#addValidationWindow').click();
  assert.equal(await page.evaluate(()=>state.validationWindows[0].id),'normal-original');
  assert.equal(await page.evaluate(()=>state.validationWindows[0].comment),'saved');
  await page.locator('#validationType').selectOption('known_abnormal');
  await page.locator('#validationStart').fill('2026-01-01T12:00');await page.locator('#validationEnd').fill('2026-01-01T13:00');
  await page.locator('#addValidationWindow').click();
  assert.equal(await page.evaluate(()=>state.validationWindows.length),1);
  await page.getByRole('button',{name:'编辑冲突窗口 normal-original'}).click();
  assert.equal(await page.locator('#addValidationWindow').getAttribute('data-editing-id'),'normal-original');
  await page.locator('#cancelValidationEdit').click();
  await page.evaluate(()=>{
   renderModelQualityStatus('failed');if(el('qualityButton').classList.contains('secondary'))throw Error('failed CTA');
   renderModelQualityStatus('passed');if(!el('qualityButton').classList.contains('secondary'))throw Error('passed CTA');
   el('ablationGroup').replaceChildren(new Option('LONG_TAG_NAME_'.repeat(12),'long'));
  });
  for(const width of [1600,1440,1366,1280,1050,900,760,390]) {
   await page.setViewportSize({width,height:1000});
   for(const stage of ['configPanel','candidatePanel','modelPanel','validationPanel','releasePanel']) {
    await page.evaluate(id=>showWorkflowStage(id),stage);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0,`${width} ${stage}`);
   }
   await page.evaluate(()=>{showWorkflowStage('candidatePanel');document.querySelector('[data-panel="stateExplorationPanel"]').click();});
   const color=await page.locator('#explorationPcColorMode').evaluate(e=>({width:e.clientWidth,visible:!!e.getClientRects().length}));
   if(color.visible)assert(color.width>=200,JSON.stringify({width,color}));
  }
  assert.deepEqual(errors,[]);
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=120,
                                env={**os.environ, "STAGE_A_URL": f"http://127.0.0.1:{server.server_port}"})
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
