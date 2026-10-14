// Precise defect hunt: svg aspect mismatch, empty bordered wrappers, header alignment.
const fs = require('fs');

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
    renderTrend(state.inspection || null);
    renderPreprocessingPreview?.({});
  }, fixture).catch(() => {});
  await page.evaluate(() => document.fonts.ready);

  const out = {};
  // 1) svg host aspect mismatch everywhere
  await page.evaluate(() => {
    document.querySelectorAll('.panel, .candidate-tool-panel, .inner-panel').forEach(p => p.classList.add('active'));
    ['modelContent', 'validationContent', 'releaseContent', 'explorationContent', 'clusterContent', 'performanceContent', 'preprocessingPreview']
      .forEach(id => { const e = document.getElementById(id); if (e) { e.hidden = false; e.style.display = ''; } });
  });
  out.svgHosts = await page.evaluate(() => [...document.querySelectorAll('svg')].map(svg => {
    const host = svg.closest('.chart,.dp-chart,.trend-chart,.preprocessing-preview-chart,.variance,.exploration-timeline,.dp-scatter-chart,.dp-trend-stat-card');
    if (!host) return null;
    const b = host.getBoundingClientRect();
    const s = svg.getBoundingClientRect();
    if (!s.width || !s.height) return null;
    return {
      host: host.id || host.className.toString().slice(0, 26),
      hostBox: [Math.round(b.width), Math.round(b.height)],
      svgBox: [Math.round(s.width), Math.round(s.height)],
      viewBox: svg.getAttribute('viewBox'),
      cssH: getComputedStyle(svg).height,
      waste: +(1 - (s.width / s.height) / ((svg.getAttribute('viewBox') || '').split(/\s+/)[2] / (svg.getAttribute('viewBox') || '').split(/\s+/)[3])).toFixed(3),
    };
  }).filter(Boolean));
  // 2) empty/near-empty bordered wrappers
  out.hairlines = await page.evaluate(() => [...document.querySelectorAll('.table-wrap, .chart, .empty')]
    .filter(n => n.getClientRects().length && n.getBoundingClientRect().height <= 6)
    .map(n => ({
      id: n.id || n.className.toString().slice(0, 30),
      h: +n.getBoundingClientRect().height.toFixed(1),
      empty: n.children.length === 0 && !n.textContent.trim(),
      inner: n.innerHTML.slice(0, 60),
    })));
  // 3) header vs main left alignment
  out.align = await page.evaluate(() => ({
    header: Math.round(document.querySelector('header h1').getBoundingClientRect().left),
    mainFirst: Math.round(document.querySelector('main > section').getBoundingClientRect().left),
    sidebarW: Math.round(document.querySelector('.workflow-sidebar').getBoundingClientRect().width),
    resultsLeft: Math.round(document.querySelector('.results').getBoundingClientRect().left),
  }));
  // 4) any element wider than its parent (overflow)
  out.overflow = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll('body *').forEach(n => {
      const p = n.parentElement;
      if (!p || !n.getClientRects().length) return;
      const nr = n.getBoundingClientRect(), pr = p.getBoundingClientRect();
      if (pr.width && nr.width - pr.width > 2 && getComputedStyle(p).overflowX === 'visible') {
        bad.push({ child: (n.id || n.className.toString()), w: Math.round(nr.width), p: (p.id || p.className.toString()), pw: Math.round(pr.width) });
      }
    });
    return bad.slice(0, 25);
  });
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
