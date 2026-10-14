// Final acceptance sweep: five stages + both dialogs at 1366x768 and 1920x1080.
const fs = require('fs');

(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const errors = [];
  const out = {};
  for (const [w, h] of [[1366, 768], [1920, 1080]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    page.on('pageerror', e => errors.push(w + ': ' + e.message));
    await page.goto(process.env.AUDIT_URL);
    const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
    await page.evaluate(f => {
      state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
      state.selectedModelTags = new Set(Object.keys(f.trained.tag_stats || {}));
      state.candidateWindows = [{ id: 'c1', source: 'trend', start: f.start, end: f.end, comment: '基准运行', enabled: true, status: 'pending' }];
      state.trainingWindows = [{ id: 't1', source: 'c1', start: f.start, end: f.end, comment: '基准', enabled: true, status: 'active' }];
      state.validationWindows = [{ id: 'v1', type: 'normal_validation', start: f.start, end: f.end, comment: '独立验证期', enabled: true }];
      renderTraining(f.trained); renderCandidateWindows(); renderTrainingWindows(); renderValidationWindows();
    }, fixture);
    await page.evaluate(() => document.fonts.ready);

    const stages = {};
    for (const stage of ['configPanel', 'candidatePanel', 'modelPanel', 'validationPanel', 'releasePanel']) {
      await page.evaluate(id => showWorkflowStage(id), stage);
      await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      stages[stage] = await page.evaluate(() => {
        const vis = n => n.getClientRects().length > 0;
        const panel = document.querySelector('.panel.active');
        const panelW = panel.getBoundingClientRect().width;
        const stretched = [...panel.querySelectorAll('button, .download')].filter(vis)
          .filter(b => b.getBoundingClientRect().width > Math.max(320, 0.45 * panelW))
          .map(b => (b.id || b.textContent.trim().slice(0, 12)) + ':' + Math.round(b.getBoundingClientRect().width));
        // sibling heading edges inside every nested group
        const ragged = [];
        panel.querySelectorAll('.group, .validation-box').forEach(g => {
          const edges = [...g.querySelectorAll(':scope > h3, :scope > h4, :scope > .group-title, :scope > .chart-card > h3, :scope > .chart-card > h4, :scope > section > h3, :scope > section > h4')]
            .filter(vis).map(n => ({ t: n.textContent.trim().slice(0, 12), l: Math.round(n.getBoundingClientRect().left) }));
          if (!edges.length) return;
          const uniq = [...new Set(edges.map(e => e.l))];
          if (uniq.length > 1) ragged.push({ group: (g.id || g.className.toString().slice(0, 20)), edges });
        });
        // chart dead space
        const charts = [...panel.querySelectorAll('.chart, .dp-chart, .trend-chart')].filter(vis).map(c => {
          const svg = c.querySelector('svg');
          const cb = c.getBoundingClientRect();
          if (!svg) return { id: c.id, box: [Math.round(cb.width), Math.round(cb.height)], svg: null };
          const sb = svg.getBoundingClientRect();
          const vb = (svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
          const waste = vb.length === 4 ? +(1 - (sb.width / sb.height) / (vb[2] / vb[3])).toFixed(3) : null;
          return { id: c.id, waste };
        }).filter(c => c.svg !== null || c.box[1] > 8);
        const overscroll = [...panel.querySelectorAll('.table-wrap')].filter(vis)
          .filter(t => t.scrollWidth > t.clientWidth + 1 && getComputedStyle(t).overflowX !== 'auto')
          .map(t => (t.id || t.className.toString().slice(0, 20)));
        return {
          pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          stretched, ragged, wasteCharts: charts.filter(c => c.waste !== null && Math.abs(c.waste) > 0.12),
          overscroll,
        };
      });
    }

    // groupProfileDialog geometry
    const profile = await page.evaluate(() => {
      const d = document.getElementById('groupProfileDialog');
      d.showModal();
      const r = d.getBoundingClientRect();
      const head = d.querySelector('header').getBoundingClientRect();
      const close = d.querySelector('#groupProfileDialogClose').getBoundingClientRect();
      const title = d.querySelector('h2').getBoundingClientRect();
      const box = { w: Math.round(r.width), h: Math.round(r.height), overflow: d.scrollWidth - d.clientWidth, titleRight: Math.round(title.right), closeLeft: Math.round(close.left), headW: Math.round(head.width) };
      d.close();
      return box;
    });

    // exceedance dialog geometry + pressed-state legibility
    await page.evaluate(() => { showWorkflowStage('modelPanel'); document.getElementById('modelContent').hidden = false; });
    await page.locator('#trainingExceedanceButton').click();
    await page.waitForFunction(() => document.querySelector('#exceedanceStatus').textContent.length > 0);
    const exc = await page.evaluate(() => {
      const d = document.getElementById('trainingExceedanceDialog');
      const btns = [...d.querySelectorAll('.exceedance-modes button')].map(b => {
        const cs = getComputedStyle(b);
        return { id: b.id, pressed: b.getAttribute('aria-pressed'), border: cs.borderTopColor, color: cs.color };
      });
      const groups = [...d.querySelectorAll('.exceedance-compare-group')].filter(g => g.getClientRects().length);
      return {
        overflowX: d.scrollWidth - d.clientWidth,
        w: Math.round(d.getBoundingClientRect().width),
        btns, distinct: new Set(btns.map(b => b.border)).size,
        groups: groups.length, groupOverflow: groups.filter(g => g.scrollWidth > g.clientWidth + 1).length,
      };
    });
    await page.keyboard.press('Escape');
    out[w + 'x' + h] = { stages, profile, exc };
    await page.close();
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
