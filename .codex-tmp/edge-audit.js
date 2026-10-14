// Compares left edges of every visible heading/control across the five stages.
const fs = require('fs');
(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    renderTraining(f.trained);
    const e = document.getElementById('modelContent'); if (e) e.hidden = false;
    const v = document.getElementById('validationContent'); if (v) v.hidden = false;
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  const out = {};
  for (const stage of ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel']) {
    await page.evaluate(id => showWorkflowStage(id), stage);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[stage] = await page.evaluate(() => {
      const vis = n => n.getClientRects().length > 0;
      const rows = [];
      document.querySelectorAll('.panel.active h3, .panel.active h4, .panel.active > * > h3, .panel.active button').forEach(n => {
        if (!vis(n)) return;
        const r = n.getBoundingClientRect();
        rows.push({ t: n.tagName, txt: n.textContent.trim().slice(0, 16), left: Math.round(r.left) });
      });
      return rows;
    });
  }
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
