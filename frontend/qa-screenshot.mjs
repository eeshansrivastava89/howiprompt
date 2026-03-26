// Capture before/after screenshots for visual QA against live server
import { chromium } from 'playwright';
import { join } from 'path';

const PHASE = process.argv[2] || 'before';
const BASE = process.argv[3] || 'http://localhost:60666';
const OUT = join(import.meta.dirname, '..', 'qa-screenshots');

const browser = await chromium.launch();

async function screenshot(name, fn) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await fn(page);
  await page.screenshot({ path: join(OUT, `${PHASE}-${name}.png`), fullPage: false });
  console.log(`  ✓ ${name}`);
  await page.close();
}

// ── Dashboard ──

await screenshot('dashboard-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
  });
  await page.waitForTimeout(500);
});

await screenshot('dashboard-dark', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.add('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
  });
  await page.waitForTimeout(500);
});

// ── Methodology modal ──

await screenshot('methodology-s1-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    window.openMethodology?.();
  });
  await page.waitForTimeout(800);
});

await screenshot('methodology-s1-dark', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.add('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    window.openMethodology?.();
  });
  await page.waitForTimeout(800);
});

// Methodology modal - scroll to S2 (tasting notes)
await screenshot('methodology-s2-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    window.openMethodology?.();
  });
  await page.waitForTimeout(600);
  await page.evaluate(() => {
    const scroll = document.getElementById('methScroll');
    const s2 = scroll?.querySelectorAll('[data-meth-section]')[1];
    s2?.scrollIntoView({ behavior: 'instant' });
  });
  await page.waitForTimeout(500);
});

// Methodology modal - scroll to S3 (personas)
await screenshot('methodology-s3-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    window.openMethodology?.();
  });
  await page.waitForTimeout(600);
  await page.evaluate(() => {
    const scroll = document.getElementById('methScroll');
    const s3 = scroll?.querySelectorAll('[data-meth-section]')[2];
    s3?.scrollIntoView({ behavior: 'instant' });
  });
  await page.waitForTimeout(500);
});

// ── Leaderboard tab ──

await screenshot('leaderboard-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab').forEach(t => {
      if (t.textContent.trim() === 'Leaderboard') t.click();
    });
  });
  await page.waitForTimeout(400);
});

await screenshot('leaderboard-dark', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.add('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab').forEach(t => {
      if (t.textContent.trim() === 'Leaderboard') t.click();
    });
  });
  await page.waitForTimeout(400);
});

// ── Setup wizard ──

await screenshot('wizard-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => document.documentElement.classList.remove('dark'));
  await page.waitForTimeout(400);
});

await screenshot('wizard-dark', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => document.documentElement.classList.add('dark'));
  await page.waitForTimeout(400);
});

// ── Settings modal ──

await screenshot('settings-light', async (page) => {
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
    window.openSettings?.();
  });
  await page.waitForTimeout(500);
});

// ── Wrapped page ──

await screenshot('wrapped-light', async (page) => {
  await page.goto(`${BASE}/wrapped/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => document.documentElement.classList.remove('dark'));
  await page.waitForTimeout(500);
});

await screenshot('wrapped-dark', async (page) => {
  await page.goto(`${BASE}/wrapped/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => document.documentElement.classList.add('dark'));
  await page.waitForTimeout(500);
});

// ── Mobile viewport ──

await screenshot('mobile-light', async (page) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(`${BASE}/index.html`, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.evaluate(() => {
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.wizard-overlay').forEach(el => el.style.display = 'none');
  });
  await page.waitForTimeout(400);
});

await browser.close();
console.log(`\n✓ ${PHASE}: all screenshots saved to qa-screenshots/`);
