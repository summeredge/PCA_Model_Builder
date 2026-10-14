const fs = require('fs');
(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => { state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained; renderTraining(f.trained); }, fixture);
  const out = await page.evaluate(() => {
    const path = n => { const parts = []; let cur = n; while (cur && cur !== document.body) { parts.unshift((cur.id || cur.className.toString().split(' ')[0] || cur.tagName).slice(0, 22)); cur = cur.parentElement; } return parts.join(' > '); };
    const chain = id => { const n = document.getElementById(id); return n ? { id, path: path(n), parent: (n.parentElement.id || n.parentElement.className.toString().slice(0, 30)), idx: [...n.parentElement.children].indexOf(n) } : null; };
    const group = document.querySelector('#modelPanel .training-configuration');
    return {
      exceed: chain('trainingExceedanceButton'),
      modelContent: chain('modelContent'),
      finalReview: chain('finalTrainingReviewTitle'),
      composition: chain('trainingCompositionReview'),
      groupChildren: group ? [...group.children].map(c => c.tagName + '.' + (c.className || '').toString().slice(0, 26) + '#' + (c.id || '')) : null,
      exceedComputed: (() => { const b = document.getElementById('trainingExceedanceButton'); const cs = getComputedStyle(b); return { justifySelf: cs.justifySelf, width: cs.width, parentDisplay: getComputedStyle(b.parentElement).display, parentId: b.parentElement.id }; })(),
      disabledStyles: (() => {
        const probe = document.createElement('button'); probe.disabled = true; probe.textContent = 'x'; document.body.append(probe);
        const cs = getComputedStyle(probe); const v = { opacity: cs.opacity, cursor: cs.cursor, bg: cs.backgroundColor, color: cs.color };
        probe.remove(); return v;
      })(),
    };
  });
  const align = {};
  for (const width of [1366, 1600, 1920, 2560]) {
    await page.setViewportSize({ width, height: 900 });
    align[width] = await page.evaluate(() => ({
      headerH1: Math.round(document.querySelector('header h1').getBoundingClientRect().left),
      mainLeft: Math.round(document.querySelector('main > section').getBoundingClientRect().left),
      resultsLeft: Math.round(document.querySelector('.results').getBoundingClientRect().left),
      cardLeft: Math.round(document.querySelector('.panel.active .chart-card, .panel.active .group')?.getBoundingClientRect().left || 0),
    }));
  }
  console.log(JSON.stringify({ ...out, align }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
