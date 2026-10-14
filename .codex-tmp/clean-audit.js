// No forced reveal: exactly what a user sees in the normal flow.
const fs = require('fs');
(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    state.validationWindows = [{ id: 'v1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true }];
    renderTraining(f.trained); renderValidationWindows();
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  const out = {};
  for (const stage of ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel']) {
    await page.evaluate(id => showWorkflowStage(id), stage);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[stage] = await page.evaluate(() => {
      const vis = n => n.getClientRects().length > 0;
      const hairlines = [...document.querySelectorAll('.table-wrap, .empty, .chart, .status, .notice, .variance')]
        .filter(n => vis(n) && n.getBoundingClientRect().height <= 8)
        .map(n => ({ id: n.id || n.className.toString().slice(0, 26), h: +n.getBoundingClientRect().height.toFixed(1) }));
      const empties = [...document.querySelectorAll('.empty')].filter(vis).map(n => ({ id: n.id || n.className.toString().slice(0, 24), h: Math.round(n.getBoundingClientRect().height), text: n.textContent.trim().slice(0, 20) }));
      const radius = {};
      ['status', 'validationEvidenceStatus', 'validationDecisionStatus', 'modelQualityStatus'].forEach(id => {
        const n = document.getElementById(id);
        if (n && vis(n)) radius[id] = getComputedStyle(n).borderRadius;
      });
      return { hairlines, empties, radius };
    });
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
