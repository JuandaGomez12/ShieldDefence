// Compara el núcleo en JavaScript con las salidas de Python (make_reference.py).
// Uso: node tests/web/parity.mjs referencia.json web/model
import { readFileSync } from "node:fs";
import { forModel, detect, buildPrompt, BiLSTM, Combiner, recommendation, needsReview, fixOcrLinks } from "../../docs/js/shield-core.js";

const [ref, dir] = process.argv.slice(2);
const cases = JSON.parse(readFileSync(ref, "utf8"));
const meta = JSON.parse(readFileSync(`${dir}/lstm_web.json`, "utf8"));
const bin = readFileSync(`${dir}/lstm_web.bin`);
const lstm = new BiLSTM(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

const comb = meta.combinador ? new Combiner(meta.combinador) : null;
let bad = { model: 0, keys: 0, prompt: 0, verdict: 0, combinador: 0, recomendacion: 0, revisar: 0, ocr: 0 }, maxDiff = 0, maxComb = 0;
for (const [i, c] of cases.entries()) {
  if (c.ocr) {
    for (const [inp, py] of c.ocr) if (fixOcrLinks(inp) !== py) { bad.ocr++; console.log("ocr", JSON.stringify(inp), fixOcrLinks(inp), "| py", py); }
    continue;
  }
  const m = forModel(c.text);
  if (m !== c.model) { if (bad.model++ < 3) console.log("model", i, JSON.stringify(m.slice(0, 200)), "\n   py", JSON.stringify(c.model.slice(0, 200))); }
  const sig = detect(c.text);
  const keys = sig.map((s) => s.key).join(","), pk = c.keys.join(",");
  if (keys !== pk) { if (bad.keys++ < 5) console.log("keys", i, keys, "| py", pk); }
  const p = lstm.predict(c.text);
  maxDiff = Math.max(maxDiff, Math.abs(p - c.prob));
  if ((p >= 0.5) !== (c.prob >= 0.5)) bad.verdict++;
  if (comb && c.comb !== null) {
    const pc = comb.prob(c.text, c.prob, sig);  // misma probabilidad de la LSTM que Python, para aislar el combinador
    maxComb = Math.max(maxComb, Math.abs(pc - c.comb));
    if (Math.abs(pc - c.comb) > 1e-6) bad.combinador++;
  }
  if (c.review !== undefined && needsReview(sig, false) !== c.review) bad.revisar++;
  for (const [i2, ph] of [[0, true], [1, false]]) {
    if (c.rec && c.rec[i2] !== null && recommendation(c.text, sig, ph) !== c.rec[i2]) {
      if (bad.recomendacion++ < 2) console.log("rec", i, ph, "\n js", recommendation(c.text, sig, ph), "\n py", c.rec[i2]);
    }
  }
  if (buildPrompt(c.text, c.prob >= 0.5, c.prob, sig) !== c.prompt) { if (bad.prompt++ < 2) console.log("prompt", i); }
}
console.log(`${cases.length - 1} casos | diferencias:`, bad, "| máx. diferencia de probabilidad:", maxDiff.toExponential(2), "| combinador:", maxComb.toExponential(2));
process.exit(Object.values(bad).some((v) => v) ? 1 : 0);
