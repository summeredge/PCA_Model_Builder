// Verifies dialog chrome resolves to design tokens and the pressed state stays legible.
const fs = require('fs');

const rgba = value => (value.match(/[\d.]+/g) || []).map(Number);

function contrast(fg, bg) {
  const lum = ([r, g, b]) => {
    const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const a = lum(fg) + 0.05, b = lum(bg) + 0.05;
  return +(Math.max(a, b) / Math.min(a, b)).toFixed(2);
}

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
    state.selectedModelTags = new Set(Object.keys(f.trained.tag_stats || {}));
    renderTraining(f.trained);
    showWorkflowStage('modelPanel');
    document.getElementById('modelContent').hidden = false;
  }, fixture);
  await page.evaluate(() => document.fonts.ready);

  const bodyBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);

  // --- group profile dialog ---
  const profile = await page.evaluate(() => {
    const d = document.getElementById('groupProfileDialog');
    d.showModal();
    const cs = getComputedStyle(d);
    const h2 = d.querySelector('h2');
    return { border: cs.borderTopColor, borderW: cs.borderTopWidth, color: cs.color, bg: cs.backgroundColor, h2: getComputedStyle(h2).fontSize };
  });
  await page.evaluate(() => document.getElementById('groupProfileDialog').close());

  // --- exceedance dialog ---
  await page.locator('#trainingExceedanceButton').click();
  await page.waitForFunction(() => document.querySelector('#exceedanceStatus').textContent.length > 0);
  const exc = await page.evaluate(() => {
    const d = document.getElementById('trainingExceedanceDialog');
    const b = id => { const n = document.getElementById(id); const cs = getComputedStyle(n); return { border: cs.borderTopColor, color: cs.color, bg: cs.backgroundColor, pressed: n.getAttribute('aria-pressed') }; };
    const group = d.querySelector('.exceedance-compare-group');
    return {
      dialog: { border: getComputedStyle(d).borderTopColor, color: getComputedStyle(d).color, open: d.open },
      on: b('exceedanceModeSingle'),
      off: b('exceedanceModeAnomaly'),
      compare: group ? { border: getComputedStyle(group).borderTopColor } : null,
    };
  });
  await page.screenshot({ path: 'C:\\Users\\shaoy\\AppData\\Local\\Temp\\pca-dialog-exceedance.png' });
  await page.keyboard.press('Escape');

  const out = { bodyBg, profile, exc, errors };
  out.checks = {
    profileBorder: rgba(profile.border), profileTextContrast: contrast(rgba(profile.color), rgba(profile.bg)),
    excTextContrast: contrast(rgba(exc.dialog.color), rgba('rgb(255,255,255)')),
    onBorder: rgba(exc.on.border), offBorder: rgba(exc.off.border),
    pressedDistinct: exc.on.border !== exc.off.border || exc.on.color !== exc.off.color,
  };
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
