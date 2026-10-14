// Left-edge alignment of sibling content inside the training group + empty wrappers.
const fs = require('fs');

(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(process.env.AUDIT_URL);
  const fixture = JSON.parse(fs.readFileSync(process.env.AUDIT_FIXTURE, 'utf8'));
  await page.evaluate(f => {
    state.fileId = f.file_id; state.runId = f.trained.run_id; state.training = f.trained;
    renderTraining(f.trained);
    showWorkflowStage('modelPanel');
    const e = document.getElementById('modelContent'); if (e) e.hidden = false;
    const ve = document.getElementById('validationContent'); if (ve) ve.hidden = false;
  }, fixture);
  await page.evaluate(() => document.fonts.ready);
  const out = await page.evaluate(() => {
    const left = sel => { const n = document.querySelector(sel); return n ? Math.round(n.getBoundingClientRect().left) : null; };
    const group = document.querySelector('#modelPanel .training-configuration');
    const rows = [];
    if (group) {
      [...group.children].forEach(child => {
        const r = child.getBoundingClientRect();
        rows.push({ tag: child.tagName, cls: (child.className || '').toString().slice(0, 30), left: Math.round(r.left), right: Math.round(r.right), pad: getComputedStyle(child).padding, w: Math.round(r.width) });
      });
    }
    return {
      rows,
      groupLeft: group ? Math.round(group.getBoundingClientRect().left) : null,
      h3left: left('#modelPanel .training-configuration > h3'),
      titleLeft: left('#finalTrainingReviewTitle'),
      varianceLeft: left('#varianceThreshold'),
      exceedLeft: left('#trainingExceedanceButton'),
      emptyWrap: ['validationMetricDetails', 'contributionStability'].map(id => {
        const n = document.getElementById(id);
        if (!n) return null;
        const cs = getComputedStyle(n);
        return { id, h: +n.getBoundingClientRect().height.toFixed(2), border: cs.borderTopWidth + ' ' + cs.borderTopColor, children: n.children.length, text: n.textContent.trim().slice(0, 12) };
      }),
    };
  });
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
