/* El LLM explicador corre en este worker para no congelar la página.
   Usa Transformers.js con el modelo ONNX cuantizado a 4 bits (unos 400 MB, el
   navegador lo guarda en caché después de la primera descarga). */
let generator = null;
let loading = null;
let lib = null;

async function load({ repo, transformersUrl }) {
  lib = await import(transformersUrl);
  lib.env.allowLocalModels = false;
  const webgpu = !!(self.navigator && navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null)));
  const files = new Map();
  const progress_callback = (p) => {
    if (p.status === "progress" && p.total) {
      files.set(p.file, [p.loaded, p.total]);
      let loaded = 0, total = 0;
      for (const [l, t] of files.values()) { loaded += l; total += t; }
      self.postMessage({ type: "progress", loaded, total });
    }
  };
  generator = await lib.pipeline("text-generation", repo, {
    dtype: "q4", device: webgpu ? "webgpu" : "wasm", progress_callback,
  });
  self.postMessage({ type: "ready", device: webgpu ? "webgpu" : "wasm" });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      loading = loading || load(msg);
      await loading;
    } else if (msg.type === "generate") {
      await loading;
      let text = "";
      const streamer = new lib.TextStreamer(generator.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => { text += t; self.postMessage({ type: "token", id: msg.id, text }); },
      });
      const t0 = performance.now();
      const messages = [{ role: "system", content: msg.system }, { role: "user", content: msg.prompt }];
      await generator(messages, { max_new_tokens: msg.maxTokens, do_sample: false, repetition_penalty: 1.1, streamer });
      self.postMessage({ type: "done", id: msg.id, text, ms: Math.round(performance.now() - t0) });
    }
  } catch (err) {
    loading = msg.type === "load" ? null : loading;
    self.postMessage({ type: "error", id: msg.id, message: String(err && err.message || err) });
  }
};
