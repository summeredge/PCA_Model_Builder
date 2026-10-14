// Finds every chart box whose painted viewBox aspect ratio does not match the box,
// i.e. letterboxed charts with dead space. Also screenshots each stage.
const path = require('path');
const fs = require('fs');
const os = require('os');

const outDir = path.join(os.tmpdir(), 'pca-ui-audit2');
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    state.candidateWindows = [{ id: 'c1', source: 'trend', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'pending' }];
    state.trainingWindows = [{ id: 't1', source: 'c1', start: f.start, end: f.end, comment: '基准', enabled: true, status: 'active' }];
    state.validationWindows = [{ id: 'v1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true }];
    renderTraining(f.trained);
    renderCandidateWindows(); renderTrainingWindows(); renderValidationWindows();
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const stages = ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel'];
  const report = {};
  for (const stage of stages) {
    await page.evaluate(id => showWorkflowStage(id), stage);
    // Make sure each nested tool panel that has charts is active at least once.
    await page.evaluate(() => {
      document.querySelectorAll('#candidatePanel .candidate-tool-panel').forEach(p => p.classList.add('active'));
      ['modelContent', 'validationContent', 'releaseContent', 'explorationContent', 'clusterContent', 'performanceContent']
        .forEach(id => { const e = document.getElementById(id); if (e) e.hidden = false; });
    });
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    report[stage] = await page.evaluate(() => {
      const rows = [];
      document.querySelectorAll('.chart, .dp-chart, .trend-chart, .variance, .exploration-timeline, .dp-scatter-chart').forEach(host => {
        if (!host.getClientRects().length) return;
        const hb = host.getBoundingClientRect();
        const svg = host.querySelector('svg');
        if (!svg) { rows.push({ id: host.id, box: [Math.round(hb.width), Math.round(hb.height)], svg: null }); return; }
        const sb = svg.getBoundingClientRect();
        const vb = (svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
        const boxRatio = +(sb.width / sb.height).toFixed(3);
        const vbRatio = vb.length === 4 ? +(vb[2] / vb[3]).toFixed(3) : null;
        const waste = vbRatio ? Math.round(Math.abs(1 - vbRatio / boxRatio) * 100) : null;
        rows.push({
          id: host.id, hostBox: [Math.round(hb.width), Math.round(hb.height)],
          svgBox: [Math.round(sb.width), Math.round(sb.height)], viewBox: svg.getAttribute('viewBox'),
          boxRatio, vbRatio, wastePct: waste,
        });
      });
      return rows;
    });
    await page.screenshot({ path: path.join(outDir, `full-${stage}.png`), fullPage: true });
  }
  report.pageErrors = errors;
  console.log(JSON.stringify(report, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
