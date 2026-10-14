(async () => {
  const { chromium } = require(process.env.AUDIT_PLAYWRIGHT);
  const browser = await chromium.launch({ headless: true, executablePath: process.env.WEB_GEOMETRY_BROWSER });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  await page.goto(process.env.AUDIT_URL);
  await page.evaluate(() => document.fonts.ready);
  const out = {};
  for (const width of [1280, 1366, 1600, 1799, 1800, 1920, 2560]) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[width] = await page.evaluate(() => ({
      h1: Math.round(document.querySelector('header h1').getBoundingClientRect().left),
      main: Math.round(document.querySelector('main > section').getBoundingClientRect().left),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));
  }
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
