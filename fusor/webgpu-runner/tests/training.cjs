// Release browser regression: build with `dx serve --release --features training-checks`.
// Run with Playwright available via NODE_PATH and CHROME pointing to Chrome.
// Only this isolated test browser masks features / injects compilation failure.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

async function instrument(page, mask) {
  await page.addInitScript(mask => {
    const audit = window.trainingAudit = { requests: [], matrix: 0, generalMatrix: 0, subgroup: 0, errors: [] };
    const features = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, 'features').get;
    Object.defineProperty(GPUAdapter.prototype, 'features', { get() {
      return new Set([...features.call(this)].filter(f =>
        !(mask === 'no-matrix' && f === 'chromium-experimental-subgroup-matrix') &&
        !(mask === 'no-subgroups' && ['subgroups', 'chromium-experimental-subgroup-matrix'].includes(f))));
    }});
    if (mask === 'no-config') Object.defineProperty(GPUAdapterInfo.prototype, 'subgroupMatrixConfigs', { get: () => [] });
    if (mask === 'ranged-width') {
      Object.defineProperty(GPUAdapterInfo.prototype, 'subgroupMinSize', { get: () => 16 });
      Object.defineProperty(GPUAdapterInfo.prototype, 'subgroupMaxSize', { get: () => 64 });
    }
    const request = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(desc) {
      audit.requests.push([...desc.requiredFeatures || []]);
      const device = await request.call(this, desc);
      device.addEventListener('uncapturederror', e => audit.errors.push(e.error.message));
      return device;
    };
    const shader = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function(desc) {
      if (desc.code.includes('subgroupMatrixMultiplyAccumulate')) {
        audit.matrix++;
        if (!desc.label?.includes('probe') && desc.label !== 'fusor fixed logical program') audit.generalMatrix++;
        if (mask === 'reject-matrix') desc = { ...desc, code: 'intentional invalid matrix shader' };
      }
      if (/subgroup(Add|Mul|Min|Max|Ballot)/.test(desc.code)) audit.subgroup++;
      return shader.call(this, desc);
    };
  }, mask);
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME,
    headless: true,
    chromiumSandbox: true,
    args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  });
  try {
    const cases = process.argv.slice(2);
    const modes = cases.length ? cases : ['accelerated', 'no-matrix', 'no-subgroups', 'no-config', 'ranged-width', 'reject-matrix'];
    const results = [];
    for (const mode of modes) {
      const page = await browser.newPage();
      await instrument(page, mode);
      await page.goto(process.env.FUSOR_URL || 'http://127.0.0.1:8900/#/train', { waitUntil: 'networkidle' });
      const result = await page.evaluate(async mode => {
        const script = [...document.scripts].find(s => s.type === 'module' && s.src.includes('fusor-webgpu-runner'));
        const module = await import(script.src);
        if (!module.checkTraining) throw Error('Rebuild the release app with --features training-checks');
        return JSON.parse(await module.checkTraining(32));
      }, mode);
      const audit = await page.evaluate(() => window.trainingAudit);
      console.log(JSON.stringify({ case: mode, ...result, audit }));
      assert.deepEqual(audit.errors, []);
      assert(Math.abs(result.first_loss - 4.197046) < 2e-4);
      assert(result.held_out_loss > 0 && result.held_out_loss < 3);
      if (mode === 'accelerated') {
        assert(audit.matrix > 0 && audit.subgroup > 0);
      } else if (mode === 'reject-matrix') {
        assert(audit.matrix > 0, 'a rejected matrix shader must still have been attempted');
      } else {
        assert.equal(audit.matrix, 0);
      }
      results.push(result);
      await page.close();
    }
    for (const result of results.slice(1)) {
      for (let i = 0; i < result.losses.length; i++) assert(Math.abs(result.losses[i] - results[0].losses[i]) < 2e-4, `training loss mismatch at case ${results.indexOf(result)}`);
      assert(Math.abs(result.held_out_loss - results[0].held_out_loss) < 2e-4);
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
