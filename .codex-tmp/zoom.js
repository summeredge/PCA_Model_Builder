// Element-level screenshots for close visual inspection.
const path = require('path');
const fs = require('fs');
const os = require('os');

const outDir = path.join(os.tmpdir(), 'pca-ui-zoom');
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    state.trainingWindows = [{ id: 't1', source: 'c1', start: f.start, end: f.end, comment: '基准', enabled: true, status: 'active' }];
    renderTraining(f.trained);
    renderTrainingWindows();
    showWorkflowStage('modelPanel');
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const shots = [
    ['t2', '#t2Chart'],
  ];
  for (const [name, sel] of shots) {
    const el = await page.$(sel);
    if (!el) { console.log('missing ' + sel); continue; }
    await el.screenshot({ path: path.join(outDir, name + '.png') });
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
