import * as ort from "https://deno.land/x/onnx_runtime@0.0.3/mod.ts";
import { logMelSpectogram, decode, trimOrPad, resample } from "./utils.ts";
import vocab from "../assets/vocab_en.json" assert { type: "json" };
import melFilters from "../assets/mel_filters.json" assert { type: "json" };

const t0 = Date.now();
if (
  await Deno.writeFile("./test", new Uint8Array([0, 0]))
    .then(() => false)
    .catch((e) => console.log(e) ?? true)
)
  setInterval(() => console.log("load", (Date.now() - t0) / 1000), 100);
const full = await ort.InferenceSession.create(
  new URL(
    "https://pub-10e35d3e9dcf488ebc5a30272db639a4.r2.dev/whisper_tiny_en_20_tokens.ort"
  ).href
);
console.log("loaded", (Date.now() - t0) / 1000);

console.log(full);

const createWhisper = async ({ sampleRate = 8000 }) => {
  return async (bytes: Uint8Array) => {
    const pcm = Float32Array.from(
      resample(new Int16Array(bytes.buffer), sampleRate, 16000),
      (x) => x / 32768.0
    );
    const mel = trimOrPad(logMelSpectogram(melFilters, pcm, pcm.length), 3000);
    const { 23939: { data = [] } = {} } = await full.run({
      mel: new ort.Tensor("float32", mel.data, [1, mel.nMel, mel.nLen]),
    });
    return decode(
      data,
      Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]))
    );
  };
};

export default () => new Response(`Hello, from Deno v${Deno.version.deno}!`);
