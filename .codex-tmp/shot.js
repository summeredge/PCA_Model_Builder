const fs = require('fs');
const path = require('path');
const os = require('os');
const outDir = path.join(os.tmpdir(), 'pca-ui-shot');
fs.mkdirSync(outDir, { recursive: true });
(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 1000 } });
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    renderTraining(f.trained);
    showWorkflowStage('modelPanel');
    const e = document.getElementById('modelContent'); if (e) e.hidden = false;
  }, fixture);
  for (const width of [1366, 1920]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.evaluate(() => document.fonts.ready);
    for (const [name, sel] of [['t2', '#t2Chart'], ['spe', '#speChart']]) {
      const el = await page.$(sel);
      await el.screenshot({ path: path.join(outDir, `${name}-${width}.png`) });
    }
    const info = await page.evaluate(() => {
      const s = document.querySelector('#t2Chart svg'), b = s.getBoundingClientRect();
      const vb = s.getAttribute('viewBox').split(/\s+/).map(Number);
      return { box: [Math.round(b.width), Math.round(b.height)], vb, scale: +(Math.min(b.width / vb[2], b.height / vb[3])).toFixed(3) };
    });
    console.log(width, JSON.stringify(info));
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
