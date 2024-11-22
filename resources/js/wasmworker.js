// CPU Player in WASM.

let URL_PREFIX = "";
let initialized = false;

onmessage = async (e) => {
    if (!initialized) {
        URL_PREFIX = e.data.urlPrefix;
        initWASM();
        initialized = true;
        return;
    }
    const result = await makeCPUMove(e.data.gameId);
    postMessage(result);
}

async function makeCPUMove(gameId) {
    const response = await fetch(`${URL_PREFIX}/state/${gameId}`);
    const gameStateResponse = await response.json()
    let suggestMoveResult = goWasmSuggestMove(JSON.stringify({
        encodedGameState: gameStateResponse.encodedGameState,
        maxThinkTimeMillis: 3000,
    }));
    if (!suggestMoveResult) {
        console.log("CPU did not find a move.");
        return null;
    }
    return JSON.parse(suggestMoveResult);
}

async function initWASM() {
    // Go compiler generated JS-to-Go glue code.
    importScripts(`${URL_PREFIX}/static/js/wasm_exec.js`);
    const go = new Go();
    // Load WASM module. Avoid caching by adding a timestamp.
    const result = await WebAssembly.instantiateStreaming(fetch(`${URL_PREFIX}/static/wasm/hexz.wasm`), go.importObject);
    const wasm = result.instance;
    // This will make the goWasmSuggestMove function globally available.
    go.run(wasm);
}
