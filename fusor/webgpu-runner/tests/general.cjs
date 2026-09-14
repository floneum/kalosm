// Opt-in release conformance exports; same CHROME/NODE_PATH setup as training.cjs.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME, headless: true, chromiumSandbox: true,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  try {
    const page = await browser.newPage();
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
    const filters = process.argv.slice(2);
    for (const filter of filters.length ? filters : ['matmul::matmul', 'matmul::mat_mul_rank', 'mat_mul_transposed_rhs', 'wide_n_columns', 'qkv_projection_triple', 'normalization', 'attention_rope']) {
      console.log(await page.evaluate(async filter => {
        const script = [...document.scripts].find(s => s.type === 'module' && s.src.includes('fusor-webgpu-runner'));
        const module = await import(script.src);
        if (!module.checkConformance) throw Error('Rebuild with --features training-checks');
        return module.checkConformance(filter);
      }, filter));
      assert.deepEqual(await page.evaluate(() => window.gpuErrors), []);
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
