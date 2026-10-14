// One-off layout audit harness (not part of the product build).
const path = require('path');
const fs = require('fs');
const os = require('os');
const PW = process.env.AUDIT_PLAYWRIGHT;
const out = path.join(os.tmpdir(), 'pca-ui-audit');
fs.mkdirSync(out, { recursive: true });

(async () => {
  const { chromium } = require(PW);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.env.AUDIT_URL);
  await page.evaluate(() => document.fonts.ready);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id;
    state.runId = f.trained.run_id;
    state.training = f.trained;
    state.tags = Object.keys(f.trained.tag_stats || {});
    state.candidateWindows = [
      { id: 'candidate-1', source: 'trend', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'pending' },
      { id: 'candidate-2', source: 'manual', start: f.start, end: f.end, comment: '负荷波动', enabled: true, status: 'accepted' },
    ];
    state.trainingWindows = [
      { id: 'train-1', source: 'candidate-1', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'active' },
    ];
    state.validationWindows = [
      { id: 'valid-1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true },
      { id: 'valid-2', type: 'known_abnormal', start: f.start, end: f.end, comment: '已知异常', enabled: true },
    ];
    renderTraining(f.trained);
    renderCandidateWindows();
    renderTrainingWindows();
    renderValidationWindows();
    showWorkflowStage('modelPanel');
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  const stages = ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel'];
  const report = {};
  for (const width of [1366, 1920]) {
    await page.setViewportSize({ width, height: width === 1366 ? 768 : 1080 });
    for (const stage of stages) {
      await page.evaluate(id => showWorkflowStage(id), stage);
      await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      report[width + '-' + stage] = await page.evaluate(() => {
        const cs = getComputedStyle;
        const visible = node => node.getClientRects().length > 0;
        const root = document.querySelector('.panel.active') || document.body;
        const buttons = [...root.querySelectorAll('button, .download')].filter(visible);
        const stretched = buttons
          .filter(b => b.parentElement && cs(b).flexGrow !== '0' && b.getBoundingClientRect().width > 320)
          .map(b => (b.id || b.textContent.slice(0, 14)));
        const groups = [...root.querySelectorAll('.group, .validation-box, .exploration-controls, .chart-card, section')].filter(visible);
        const nested = groups.filter(g => {
          const p = g.parentElement && g.parentElement.closest('.group, .validation-box, .chart-card');
          return !!p && visible(p);
        }).map(g => String(g.id || g.className).slice(0, 44));
        const headings = [...root.querySelectorAll('h1,h2,h3,h4')].filter(visible)
          .map(h => h.tagName + '|' + h.textContent.trim().slice(0, 16) + '|' + cs(h).fontSize + '|' + cs(h).fontWeight);
        const charts = [...root.querySelectorAll('.chart, .dp-chart, .trend-chart')].filter(visible).map(c => {
          const r = c.getBoundingClientRect();
          const svg = c.querySelector('svg');
          const sr = svg ? svg.getBoundingClientRect() : null;
          const plot = svg ? svg.querySelector('rect') : null;
          const pr = plot ? plot.getBoundingClientRect() : null;
          return {
            id: c.id,
            w: Math.round(r.width),
            h: Math.round(r.height),
            svgH: sr ? Math.round(sr.height) : null,
            plotH: pr ? Math.round(pr.height) : null,
            slack: sr && pr ? Math.round(sr.height - pr.height) : null,
            viewBox: svg ? svg.getAttribute('viewBox') : null,
          };
        });
        const tables = [...root.querySelectorAll('.table-wrap')].filter(visible).map(t => ({
          id: t.id,
          w: Math.round(t.getBoundingClientRect().width),
          over: t.scrollWidth - t.clientWidth,
          sh: t.scrollHeight,
          ch: t.clientHeight,
          maxH: cs(t).maxHeight,
        }));
        const sec = document.querySelector('.results');
        const items = [...(document.querySelector('.panel.active')?.children || [])].filter(visible);
        let filled = 0;
        items.forEach(i => {
          const b = i.getBoundingClientRect();
          if (b.top < innerHeight) filled = Math.max(filled, Math.min(b.bottom, innerHeight) - Math.max(b.top, 0));
        });
        return {
          stage: root.id,
          stretched,
          nestedCount: nested.length,
          nested: nested.slice(0, 8),
          headings: headings.slice(0, 18),
          charts,
          tables,
          firstScreen: {
            workspaceTop: Math.round(sec.getBoundingClientRect().top),
            usable: Math.round(innerHeight - Math.max(sec.getBoundingClientRect().top, 0)),
            filled: Math.round(filled),
          },
          buttons: buttons.length,
        };
      });
      await page.screenshot({ path: path.join(out, width + '-' + stage + '.png'), fullPage: false });
    }
  }
  report.errors = errors;
  console.log(JSON.stringify(report, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
