// Measures painted chart area vs container box to find letterbox blank space.
const path = require('path');
const fs = require('fs');
const os = require('os');

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
    renderTraining(f.trained);
    showWorkflowStage('modelPanel');
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  const out = {};
  for (const width of [1366, 1600, 1920]) {
    await page.setViewportSize({ width, height: width === 1366 ? 768 : 1080 });
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[width] = await page.evaluate(() => {
      const report = [];
      document.querySelectorAll('svg').forEach(svg => {
        const host = svg.closest('.chart, .dp-chart, .trend-chart, .variance, .dp-trend-stat-card');
        if (!host || !svg.getClientRects().length) return;
        const hb = host.getBoundingClientRect();
        const sb = svg.getBoundingClientRect();
        let box;
        try { box = svg.getBBox(); } catch { return; }
        if (!box.width || !box.height) return;
        const ctm = svg.getScreenCTM();
        const corners = [[box.x, box.y], [box.x + box.width, box.y], [box.x, box.y + box.height], [box.x + box.width, box.y + box.height]]
          .map(([x, y]) => ({ x: ctm.a * x + ctm.c * y + ctm.e, y: ctm.b * x + ctm.d * y + ctm.f }));
        const painted = {
          top: Math.min(...corners.map(c => c.y)), bottom: Math.max(...corners.map(c => c.y)),
          left: Math.min(...corners.map(c => c.x)), right: Math.max(...corners.map(c => c.x)),
        };
        report.push({
          id: host.id || host.className.toString().slice(0, 24),
          host: [Math.round(hb.width), Math.round(hb.height)],
          svg: [Math.round(sb.width), Math.round(sb.height)],
          viewBox: svg.getAttribute('viewBox'),
          blankTop: Math.round(painted.top - sb.top),
          blankBottom: Math.round(sb.bottom - painted.bottom),
          blankLeft: Math.round(painted.left - sb.left),
          blankRight: Math.round(sb.right - painted.right),
        });
      });
      const cards = [...document.querySelectorAll('#modelContent .chart-card')].filter(c => c.getClientRects().length)
        .map(c => ({ head: (c.querySelector('h3,h4')?.textContent || '').slice(0, 14), box: [Math.round(c.getBoundingClientRect().width), Math.round(c.getBoundingClientRect().height)] }));
      const grid = document.querySelector('.model-projection-grid');
      return { report, cards, gridCols: grid ? getComputedStyle(grid).gridTemplateColumns : null };
    });
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
