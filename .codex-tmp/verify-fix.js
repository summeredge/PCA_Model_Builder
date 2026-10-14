// Validates the proposed CSS fixes by injecting them at runtime and measuring.
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
    showWorkflowStage('modelPanel');
    const e = document.getElementById('modelContent'); if (e) e.hidden = false;
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const measure = () => page.evaluate(() => {
    const box = id => { const n = document.getElementById(id); if (!n) return null; const r = n.getBoundingClientRect(); const s = n.querySelector('svg'); const sr = s ? s.getBoundingClientRect() : null; return { id, h: Math.round(r.height), w: Math.round(r.width), svgH: sr ? Math.round(sr.height) : null, svgW: sr ? Math.round(sr.width) : null }; };
    const plotly = document.getElementById('explorationPcChart');
    return {
      charts: ['t2Chart', 'speChart', 'scoreChart', 'loadingChart', 'trendHistogram', 'clusterChart', 'validationT2Chart', 'validationSpeChart'].map(box),
      pc: plotly ? { h: Math.round(plotly.getBoundingClientRect().height), w: Math.round(plotly.getBoundingClientRect().width), hasCanvas: !!plotly.querySelector('.plot-container') } : null,
      exceedW: Math.round(document.getElementById('trainingExceedanceButton').getBoundingClientRect().width),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      headerH1: Math.round(document.querySelector('header h1').getBoundingClientRect().left),
      sidebarLeft: Math.round(document.querySelector('.workflow-sidebar').getBoundingClientRect().left),
    };
  });

  const before = await measure();
  // Inject candidate fixes.
  await page.addStyleTag({ content: `
    .chart { height:auto; }
    #modelContent > button { justify-self:start; }
    header { padding:16px max(24px, calc((100% - 1800px) / 2 + clamp(16px, 2.5vw, 32px))); }
  ` });
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
  const after = await measure();
  const widths = {};
  for (const w of [1366, 1600, 1920, 2560]) {
    await page.setViewportSize({ width: w, height: 900 });
    widths[w] = await page.evaluate(() => ({
      headerH1: Math.round(document.querySelector('header h1').getBoundingClientRect().left),
      mainLeft: Math.round(document.querySelector('main > section').getBoundingClientRect().left),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));
  }
  console.log(JSON.stringify({ before, after, widths, errors }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
