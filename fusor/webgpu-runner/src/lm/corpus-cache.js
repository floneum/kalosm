// Immutable corpus snapshots live outside the application bundle. Cache only
// verified bytes; denied storage still permits an ordinary online session.
export async function loadCorpus(url, size, hash) {
    try {
        let cache;
        try { cache = await caches.open("fusor-corpus-v1"); } catch (_) {}
        const valid = async bytes => bytes.byteLength === size &&
            [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
                .map(x => x.toString(16).padStart(2, "0")).join("") === hash;
        let bytes;
        const saved = await cache?.match(url);
        if (saved) {
            const candidate = await saved.arrayBuffer();
            if (await valid(candidate)) bytes = candidate;
            else await cache.delete(url);
        }
        if (!bytes) {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Corpus download returned HTTP ${response.status}`);
            bytes = await response.arrayBuffer();
            if (!await valid(bytes)) throw new Error("Corpus download failed its integrity check");
            try { await cache?.put(url, new Response(bytes)); } catch (_) {}
        }
        return new TextDecoder().decode(bytes);
    } catch (error) {
        throw `Could not load training text: ${error.message || error}. Connect to the internet and retry.`;
    }
}
