// CPU Player in WASM.

// Go compiler generated JS-to-Go glue code.
importScripts("/hexz/static/js/wasm_exec.js");

onmessage = async (e) => {
    const result = await makeCPUMove(e.data.gameId);
    postMessage(result);
}

async function makeCPUMove(gameId) {
    const response = await fetch(`/hexz/state/${gameId}`);
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
    const go = new Go();
    // Load WASM module. Avoid caching by adding a timestamp.
    const result = await WebAssembly.instantiateStreaming(fetch(`/hexz/static/wasm/hexz.wasm`), go.importObject);
    const wasm = result.instance;
    // This will make the goWasmSuggestMove function globally available.
    go.run(wasm);
}

initWASM();
