const path = require('path');
const os = require('os');
const fs = require('fs');
const outDir = path.join(os.tmpdir(), 'pca-dialog-shot');
fs.mkdirSync(outDir, { recursive: true });
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
    renderTraining(f.trained);
    showWorkflowStage('modelPanel');
    document.getElementById('modelContent').hidden = false;
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  await page.locator('#trainingExceedanceButton').click();
  await page.waitForFunction(() => document.querySelector('#exceedanceStatus').textContent.length > 0);
  await page.evaluate(() => document.getElementById('exceedanceComparePanel').hidden = false).catch(() => {});
  await page.screenshot({ path: path.join(outDir, 'exceedance.png') });
  await page.keyboard.press('Escape');
  // group profile dialog chrome only
  await page.evaluate(() => {
    const d = document.getElementById('groupProfileDialog');
    d.showModal();
    document.getElementById('groupProfileDialogBody').innerHTML = '<div class="empty">审查用占位内容。</div>';
  });
  await page.screenshot({ path: path.join(outDir, 'profile.png') });
  console.log(JSON.stringify({ errors }));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
