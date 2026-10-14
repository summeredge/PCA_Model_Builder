// Every button whose box is stretched far beyond its content, with everything revealed.
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
    ['modelContent', 'validationContent', 'releaseContent', 'frozenReplay', 'modelQualityResults', 'explorationContent', 'clusterContent', 'performanceContent', 'basicInspectionIssues']
      .forEach(id => { const e = document.getElementById(id); if (e) { e.hidden = false; e.removeAttribute('hidden'); } });
    ['modelEmpty', 'validationEmpty', 'releaseEmpty', 'explorationEmpty', 'clusterEmpty', 'performanceEmpty'].forEach(id => {
      const e = document.getElementById(id); if (e) e.hidden = true;
    });
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));

  const out = await page.evaluate(() => {
    const rows = [];
    document.querySelectorAll('button, .download, a.button').forEach(b => {
      if (!b.getClientRects().length) return;
      const r = b.getBoundingClientRect();
      const parent = b.parentElement;
      const pr = parent.getBoundingClientRect();
      const cs = getComputedStyle(parent);
      if (r.width > 240 && r.width > 0.5 * pr.width) {
        rows.push({
          id: b.id || b.className + ':' + b.textContent.trim().slice(0, 12),
          w: Math.round(r.width), parentW: Math.round(pr.width),
          parent: (parent.id || parent.className.toString().slice(0, 28)),
          parentDisplay: cs.display, flexGrow: getComputedStyle(b).flexGrow, justifySelf: getComputedStyle(b).justifySelf,
          inlineStyle: b.getAttribute('style') || '',
        });
      }
    });
    // also: '.empty' placeholders and stray hairlines with everything revealed
    const hairlines = [...document.querySelectorAll('.table-wrap, .status, .notice, .empty, .chart')]
      .filter(n => n.getClientRects().length && n.getBoundingClientRect().height <= 8)
      .map(n => ({ id: n.id || n.className.toString().slice(0, 30), h: +n.getBoundingClientRect().height.toFixed(1) }));
    return { rows, hairlines };
  });
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
