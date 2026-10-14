// Cross-stage component consistency + chart chrome measurements.
const fs = require('fs');
const path = require('path');
const os = require('os');

const outDir = path.join(os.tmpdir(), 'pca-ui-consistency');
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
    state.candidateWindows = [{ id: 'c1', source: 'trend', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'pending' }];
    state.trainingWindows = [{ id: 't1', source: 'c1', start: f.start, end: f.end, comment: '基准', enabled: true, status: 'active' }];
    state.validationWindows = [{ id: 'v1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true }];
    renderTraining(f.trained); renderCandidateWindows(); renderTrainingWindows(); renderValidationWindows();
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const stages = ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel'];
  const style = n => {
    const cs = getComputedStyle(n);
    return {
      pad: cs.padding, gap: cs.gap, border: cs.borderWidth + ' ' + cs.borderColor, radius: cs.borderRadius,
      bg: cs.backgroundColor, font: cs.fontSize + '/' + cs.fontWeight,
    };
  };
  const rows = [];
  for (const stage of stages) {
    await page.evaluate(id => showWorkflowStage(id), stage);
    await page.evaluate(() => {
      document.querySelectorAll('#candidatePanel .candidate-tool-panel').forEach(p => p.classList.add('active'));
      ['modelContent', 'validationContent', 'releaseContent', 'explorationContent', 'clusterContent', 'performanceContent']
        .forEach(id => { const e = document.getElementById(id); if (e) e.hidden = false; });
    });
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    const data = await page.evaluate(() => {
      const vis = n => n.getClientRects().length > 0;
      const pick = (sel, limit) => [...document.querySelectorAll(sel)].filter(vis).slice(0, limit || 3)
        .map(n => ({ sel, on: (n.id || n.className).toString().slice(0, 30), style: null }));
      const grab = (sel, limit) => [...document.querySelectorAll(sel)].filter(vis).slice(0, limit || 3).map(n => {
        const cs = getComputedStyle(n);
        return { on: (n.id || n.className).toString().slice(0, 34), pad: cs.padding, gap: cs.gap, radius: cs.borderRadius, border: cs.borderWidth, font: cs.fontSize + '/' + cs.fontWeight, color: cs.color, bg: cs.backgroundColor, h: Math.round(n.getBoundingClientRect().height) };
      });
      return {
        h3: grab('.panel.active h3', 6),
        h4: grab('.panel.active h4', 4),
        group: grab('.panel.active .group', 4),
        chartCard: grab('.panel.active .chart-card', 4),
        chartTile: grab('.panel.active h3', 0) && grab(null, 0),
        buttons: grab('.panel.active button, .panel.active .download', 8),
        tableWrap: grab('.panel.active .table-wrap', 3),
        th: grab('.panel.active th', 3),
        help: grab('.panel.active .help, .panel.active .legend', 3),
        notice: grab('.panel.active .notice, .panel.active .status', 3),
      };
    });
    rows.push({ stage, data });
    await page.screenshot({ path: path.join(outDir, stage + '.png'), fullPage: false });
  }
  console.log(JSON.stringify({ rows, errors }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
