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
    const rowMode = process.env.FUSOR_ROW_MODE;
    assert(!rowMode || ['permuted-lanes', 'partial-subgroups'].includes(rowMode));
    await page.addInitScript(rowMode => {
      window.gpuErrors = [];
      window.rowShaders = 0;
      const request = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function(desc) {
        const device = await request.call(this, desc);
        device.addEventListener('uncapturederror', e => window.gpuErrors.push(e.error.message));
        return device;
      };
      if (rowMode) {
        const create = GPUDevice.prototype.createShaderModule;
        GPUDevice.prototype.createShaderModule = function(desc) {
          let code = desc.code;
          // Slab row collectives use this builtin to guard full occupancy.
          // Fixed training programs have separate capability coverage.
          if (code.includes('@builtin(num_subgroups)') && !code.includes('arena: array<u32>')) {
            window.rowShaders++;
            const builtin = rowMode === 'permuted-lanes' ? 'local_invocation_index' : 'num_subgroups';
            const match = code.match(new RegExp(`@builtin\\(${builtin}\\)\\s+(\\w+)\\s*:\\s*u32`));
            if (!match) throw Error(`missing ${builtin} in row shader`);
            const raw = `fusor_raw_${match[1]}`;
            const block = Number(code.match(/@workgroup_size\((\d+)/)[1]);
            // A permutation across subgroups must not alter logical results.
            // Inflating the subgroup count forces the workgroup-tree fallback.
            const expression = rowMode === 'permuted-lanes' ? `${raw} ^ ${block / 2}u` : `${raw} + 1u`;
            code = code.replace(match[0], match[0].replace(match[1], raw));
            code = code.replace(/(fn main\([\s\S]*?\)\s*\{)/, `$1\nlet ${match[1]} = ${expression};`);
          }
          return create.call(this, { ...desc, code });
        };
      }
    }, rowMode);
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
    if (rowMode) assert(await page.evaluate(() => window.rowShaders > 0), 'the row collective path must be exercised');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
