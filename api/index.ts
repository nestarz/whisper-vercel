import { readerFromStreamReader } from "https://deno.land/std@0.107.0/io/streams.ts";
import * as ort from "https://deno.land/x/onnx_runtime@0.0.3/mod.ts";
import { logMelSpectogram, decode, trimOrPad, resample } from "./utils.ts";
import vocab from "../assets/vocab_en.json" assert { type: "json" };
import melFilters from "../assets/mel_filters.json" assert { type: "json" };

const modelPath =
  "https://pub-10e35d3e9dcf488ebc5a30272db639a4.r2.dev/whisper_tiny_en_20_tokens.ort";

let model;
const createWhisper = async ({ sampleRate = 8000 } = {}) => {
  model = model ?? (await ort.InferenceSession.create(modelPath));

  return async (bytes: Uint8Array) => {
    console.log(bytes);
    const pcm = Float32Array.from(
      resample(new Int16Array(bytes.buffer), sampleRate, 16000),
      (x) => x / 32768.0
    );
    const mel = trimOrPad(logMelSpectogram(melFilters, pcm, pcm.length), 3000);
    const { 23939: { data = [] } = {} } = await model.run({
      mel: new ort.Tensor("float32", mel.data, [1, mel.nMel, mel.nLen]),
    });
    console.log(mel, data);
    return decode(
      data,
      Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]))
    );
  };
};

export default async (a: { request: Request }) => {
  const req = a.request;
  console.log(a.body, await a.json().catch(() => null), req.body, await req.json().catch(() => null));
  if (
    req.method !== "POST" ||
    req.headers.get("content-type") !== "application/octet-stream"
  )
    return new Response(null);
  const { sample_rate } = Object.fromEntries(new URLSearchParams(req.url));
  const options = { sampleRate: Number(sample_rate ?? 8000) };
  const whisper = await createWhisper(options);
  const r = await Deno.readAll(readerFromStreamReader(req.body.getReader()));
  console.log(r);
  const result = await whisper(new Uint8Array(r));

  return new Response(result, { headers: { "content-type": "text/plain" } });
};
