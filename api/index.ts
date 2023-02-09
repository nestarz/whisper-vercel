import * as ort from "https://deno.land/x/onnx_runtime@0.0.3/mod.ts";
import { logMelSpectogram, decode, trimOrPad, resample } from "./utils.ts";
import vocab from "../assets/vocab_en.json" assert { type: "json" };
import melFilters from "../assets/mel_filters.json" assert { type: "json" };

const modelPath =
  "https://pub-10e35d3e9dcf488ebc5a30272db639a4.r2.dev/whisper_tiny_en_20_tokens.ort";

let model;
const createWhisper = async ({ sampleRate = 8000 }) => {
  model = model ?? (await ort.InferenceSession.create(modelPath));

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

async function streamToArrayBuffer(
  stream: ReadableStream<Uint8Array>
): Promise<ArrayBuffer> {
  let result = new Uint8Array(0);
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const newResult = new Uint8Array(result.length + value.length);
    newResult.set(result);
    newResult.set(value, result.length);
    result = newResult;
  }
  return result.buffer;
}

export default async (req: Request) => {
  if (req.method !== "POST" || !req.body) return new Response(null);
  console.log(req, req.clone);

  const buffer = await streamToArrayBuffer(req.body);
  console.log(buffer);
  const { sample_rate } = Object.fromEntries(new URLSearchParams(req.url));
  const whisper = await createWhisper({ sampleRate: Number(sample_rate) });
  const result = await whisper(new Uint8Array(buffer));

  return new Response(result, { headers: { "content-type": "text/plain" } });
};
