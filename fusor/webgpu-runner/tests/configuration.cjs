// Exercise the production UI, including a non-tiled shape and a complete default run.
// NODE_PATH must expose Playwright; CHROME selects a browser with WebGPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME, headless: true, chromiumSandbox: true,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => { errors.push(error.message); console.error(error.message); });
    await page.addInitScript(() => {
      window.gpuErrors = [];
      const request = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function(desc) {
        const device = await request.call(this, desc);
        device.addEventListener('uncapturederror', e => window.gpuErrors.push(e.error.message));
        return device;
      };
    });
    await page.goto(process.env.FUSOR_URL || 'http://127.0.0.1:8900/#/train', { waitUntil: 'networkidle' });
    if (process.env.FUSOR_PRODUCTION) {
      const diagnostics = await page.evaluate(async () => {
        const script = [...document.scripts].find(s => s.type === 'module' && s.src.includes('fusor-webgpu-runner'));
        const module = await import(script.src);
        return ['checkTraining', 'checkProgram', 'checkConformance', 'checkBenchmarks'].filter(n => n in module);
      });
      assert.deepEqual(diagnostics, []);
    }
    const field = name => page.getByRole('spinbutton', { name, exact: true });
    const button = name => page.getByRole('button', { name, exact: true });
    const metric = async label => page.locator('.metric').filter({ has: page.locator('.metric-label', { hasText: new RegExp(`^${label}$`) }) }).locator('.metric-value').innerText();
    const idle = () => page.waitForFunction(() => !document.querySelector('.lm-config-fields').disabled, null, { timeout: 120000 });
    assert.equal(await field('Context tokens').inputValue(), '128');
    assert.equal(await field('Training tokens').inputValue(), '20000000');
    assert((await page.locator('body').innerText()).includes('7,999,444'));
    await field('Attention heads').fill('5');
    assert.match(await page.getByRole('alert').innerText(), /divisible/);
    assert(await button('Start training').isDisabled());
    await field('Attention heads').fill('0');
    assert.match(await page.getByRole('alert').innerText(), /between/);
    await field('Attention heads').fill('4');
    await field('Batch size').fill('128');
    await field('Context tokens').fill('512');
    assert.match(await page.getByRole('alert').innerText(), /memory/);
    for (const [name, value] of Object.entries({
      'Blocks': 2, 'Model width': 42, 'Attention heads': 3,
      'Feed-forward width': 75, 'Context tokens': 19, 'Batch size': 3, 'Training tokens': 500,
    })) await field(name).fill(String(value));
    await button('Apply configuration').click();
    assert.match(await page.locator('.lm-config-summary').innerText(), /57 tokens \/ step/);
    await button('Start training').click();
    await button('Training complete').waitFor({ timeout: 120000 });
    await idle();
    assert.equal(await metric('steps'), '9');
    assert.equal(await metric('tokens trained'), '513');
    assert(Number.isFinite(Number(await metric('train loss'))));
    await button('Read the model').click();
    await idle();
    assert.equal(await page.locator('.lm-map').count(), 6);
    assert.deepEqual(await page.locator('.lm-map').evaluateAll(imgs => imgs.map(i => i.naturalWidth)), [19, 19, 19, 19, 19, 19]);
    await button('Write 400 characters').click();
    await idle();
    assert.equal((await page.locator('.lm-sample > span').last().innerText()).length, 400);
    await button('Reset').click();
    assert.equal(await metric('steps'), '0');
    assert.equal(await page.locator('.lm-map').count(), 0);
    console.log('custom configuration, budget, generation and attention passed');

    await button('Longer context · 128').click();
    const quick = !!process.env.FUSOR_QUICK;
    if (quick) await field('Training tokens').fill('524288');
    await button('Apply configuration').click();
    const start = Date.now();
    await button('Start training').click();
    await page.waitForFunction(() => Number(document.querySelector('progress').value) >= 16, null, { timeout: 120000 });
    assert(await field('Model width').isDisabled());
    await button('Pause').click();
    await idle();
    const paused = Number(await metric('steps'));
    assert(paused >= 16);
    await button('Resume').click();
    await button('Training complete').waitFor({ timeout: 300000 });
    await idle();
    assert.equal(await metric('steps'), quick ? '256' : '9766');
    assert.equal(await metric('tokens trained'), quick ? '524,288' : '20,000,768');
    assert(Number(await metric('held-out loss')) < 2.5);
    assert.deepEqual(errors, []);
    assert.deepEqual(await page.evaluate(() => window.gpuErrors), []);
    const result = {
      wall_seconds: (Date.now() - start) / 1000,
      ms_per_step: await metric('per step'),
      loss: await metric('train loss'), held_out_loss: await metric('held-out loss'),
      accuracy: await metric('next-char top-1'), steps: await metric('steps'),
    };
    console.log(JSON.stringify(result));
    if (process.env.FUSOR_ARTIFACTS) {
      fs.mkdirSync(process.env.FUSOR_ARTIFACTS, { recursive: true });
      fs.writeFileSync(`${process.env.FUSOR_ARTIFACTS}/configuration.json`, JSON.stringify(result, null, 2));
      await page.screenshot({ path: `${process.env.FUSOR_ARTIFACTS}/configuration.png`, fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: `${process.env.FUSOR_ARTIFACTS}/configuration-mobile.png`, fullPage: true });
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
