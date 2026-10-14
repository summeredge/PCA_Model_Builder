// Primary-button competition, stretched buttons, and stray off-token colors.
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
    state.candidateWindows = [{ id: 'c1', source: 'trend', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'pending' }];
    state.trainingWindows = [{ id: 't1', source: 'c1', start: f.start, end: f.end, comment: '基准', enabled: true, status: 'active' }];
    state.validationWindows = [{ id: 'v1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true }];
    renderTraining(f.trained); renderCandidateWindows(); renderTrainingWindows(); renderValidationWindows();
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const out = {};
  for (const stage of ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel']) {
    await page.evaluate(id => showWorkflowStage(id), stage);
    await page.evaluate(() => {
      document.querySelectorAll('#candidatePanel .candidate-tool-panel').forEach(p => p.classList.add('active'));
      ['modelContent', 'validationContent', 'releaseContent'].forEach(id => { const e = document.getElementById(id); if (e) e.hidden = false; });
    });
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[stage] = await page.evaluate(() => {
      const vis = n => n.getClientRects().length > 0;
      const buttons = [...document.querySelectorAll('.panel.active button, .panel.active .download')].filter(vis);
      const primary = buttons.filter(b => !b.classList.contains('secondary') && !b.classList.contains('danger'));
      const neighbors = [];
      for (let i = 0; i < primary.length; i++) {
        for (let j = i + 1; j < primary.length; j++) {
          const a = primary[i].getBoundingClientRect(), b = primary[j].getBoundingClientRect();
          const sameRow = Math.abs(a.top - b.top) < 6 && Math.abs(a.bottom - b.bottom) < 6;
          if (sameRow && b.left - a.right < 24 && b.left >= a.left) {
            neighbors.push([primary[i].id || primary[i].textContent.slice(0, 10), primary[j].id || primary[j].textContent.slice(0, 10)]);
          }
        }
      }
      const wide = buttons.filter(b => b.getBoundingClientRect().width > 0.45 * b.closest('.panel').getBoundingClientRect().width)
        .map(b => ({ id: b.id || b.textContent.slice(0, 12), w: Math.round(b.getBoundingClientRect().width), panelW: Math.round(b.closest('.panel').getBoundingClientRect().width) }));
      const colors = new Map();
      document.querySelectorAll('.panel.active *').forEach(n => {
        if (!vis(n)) return;
        const cs = getComputedStyle(n);
        [['bg', cs.backgroundColor], ['color', cs.color], ['border', cs.borderTopColor]].forEach(([k, v]) => {
          if (!v || v === 'rgba(0, 0, 0, 0)') return;
          const key = k + ':' + v;
          if (!colors.has(key)) colors.set(key, (n.id || n.className.toString().slice(0, 22)));
        });
      });
      return {
        total: buttons.length, primary: primary.map(b => b.id || b.textContent.slice(0, 14)),
        competing: neighbors, wide,
        unusual: [...colors.entries()].filter(([k]) => /rgb\((248, 250, 252|255, 248, 231|232, 237, 243|238, 242, 246|23[0-9], 24[0-9], 25[0-9])/.test(k)).slice(0, 12),
      };
    });
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
