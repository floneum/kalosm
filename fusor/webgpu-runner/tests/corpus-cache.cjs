// Verify first-download failure, retry, persistent cache and corrupt-cache recovery.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch({ executablePath: process.env.CHROME, headless: true });
    try {
        const page = await browser.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        const asset = '**/raw.githubusercontent.com/**/assets/tinystories.txt';
        const ready = () => page.getByRole('spinbutton', { name: 'Context tokens', exact: true }).waitFor({ timeout: 60000 });
        const retry = () => page.getByRole('button', { name: 'Retry download', exact: true });
        await page.route(asset, route => route.abort());
        await page.goto(process.env.FUSOR_URL || 'http://127.0.0.1:8900/#/train');
        await retry().waitFor();
        assert.match(await page.getByRole('alert').innerText(), /Could not load training text/);
        await page.unroute(asset);
        await retry().click();
        await ready();
        assert((await page.locator('body').innerText()).includes('7,999,444'));

        await page.route(asset, route => route.abort());
        await page.reload();
        await ready(); // A new WASM instance loads without any corpus network access.

        await page.evaluate(async () => {
            const cache = await caches.open('fusor-corpus-v1');
            for (const key of await cache.keys()) await cache.put(key, new Response('corrupt cache'));
        });
        await page.reload();
        await retry().waitFor(); // Corrupt bytes must never reach the tokenizer/model.
        await page.unroute(asset);
        await retry().click();
        await ready();
        assert.deepEqual(errors, []);
        console.log('download failure, retry, offline corpus cache and corruption recovery passed');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
