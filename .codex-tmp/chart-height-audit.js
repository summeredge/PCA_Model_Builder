// Confirms every chart host keeps a usable, sane height after the aspect fix.
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
    renderTraining(f.trained);
    document.querySelectorAll('.panel, .candidate-tool-panel, .inner-panel').forEach(p => p.classList.add('active'));
    ['modelContent', 'validationContent', 'releaseContent', 'frozenReplay', 'explorationContent', 'clusterContent', 'performanceContent'].forEach(id => { const e = document.getElementById(id); if (e) e.hidden = false; });
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
  const out = {};
  for (const width of [1366, 1920]) {
    await page.setViewportSize({ width, height: 1080 });
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[width] = await page.evaluate(() => {
      const rows = [];
      document.querySelectorAll('.chart, .dp-chart, .trend-chart, .variance, .exploration-timeline, .preprocessing-preview-chart, .dp-scatter-chart').forEach(n => {
        if (!n.getClientRects().length) return;
        const r = n.getBoundingClientRect();
        const svg = n.querySelector('svg');
        const sr = svg ? svg.getBoundingClientRect() : null;
        rows.push({
          id: n.id || n.className.toString().slice(0, 24),
          h: Math.round(r.height), w: Math.round(r.width),
          svgH: sr ? Math.round(sr.height) : null,
          plotly: !!n.querySelector('.plot-container'),
          css: getComputedStyle(n).height,
        });
      });
      return { rows, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth };
    });
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
