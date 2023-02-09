function dft(signal: Float32Array): Float32Array {
  let N = signal.length;
  let out: Float32Array = new Float32Array(N * 2);

  for (let k = 0; k < N; k++) {
    let re = 0;
    let im = 0;

    for (let n = 0; n < N; n++) {
      let angle = (2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im -= signal[n] * Math.sin(angle);
    }

    out[k * 2 + 0] = re;
    out[k * 2 + 1] = im;
  }

  return out;
}

function fft(signal: Float32Array): Float32Array {
  let N = signal.length;
  let out = new Float32Array(N * 2);

  if (N === 1) {
    out[0] = signal[0];
    out[1] = 0;
    return out;
  }

  if (N % 2 === 1) return dft(signal);

  let even = signal.filter((_, i) => i % 2 === 0);
  let odd = signal.filter((_, i) => i % 2 !== 0);

  let even_fft = fft(even);
  let odd_fft = fft(odd);

  for (let k = 0; k < N / 2; k++) {
    let theta = (2 * Math.PI * k) / N;

    let re = Math.cos(theta);
    let im = -Math.sin(theta);

    let re_odd = odd_fft[2 * k + 0];
    let im_odd = odd_fft[2 * k + 1];

    out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
    out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

    out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
    out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
  }

  return out;
}

function getWindow(windowSize: number): Float32Array {
  let window = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++)
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / windowSize));
  return window;
}

interface Mel {
  nLen: number;
  nMel: number;
  data: Float32Array;
}

const WHISPER_SAMPLE_RATE = 16000;
const WHISPER_N_FFT = 400;
const WHISPER_N_MEL = 80;
const WHISPER_HOP_LENGTH = 160;
const WHISPER_CHUNK_SIZE = 30;

export const logMelSpectogram = (
  filters_,
  samples,
  __,
  _ = WHISPER_SAMPLE_RATE,
  fftSize = WHISPER_N_FFT,
  fftStep = WHISPER_HOP_LENGTH,
  nMel = WHISPER_N_MEL,
  speedUp
) => {
  const nSamples = WHISPER_CHUNK_SIZE * WHISPER_SAMPLE_RATE;
  const filters = filters_.flat();
  const hann = getWindow(fftSize);
  const nLen = Math.floor((samples.length - fftSize) / fftStep) + 1;
  const mel: Mel = { nMel, nLen, data: new Float32Array(nMel * nLen) };
  const nFft = 1 + (speedUp ? fftSize / 4 : fftSize / 2);
  const fftIn = new Float32Array(fftSize).fill(0.0);
  for (let i = 0; i < mel.nLen; i++) {
    const offset = i * fftStep;
    // apply Hanning window
    for (let j = 0; j < fftSize; j++) {
      if (offset + j < nSamples) fftIn[j] = hann[j] * samples[offset + j];
      else fftIn[j] = 0.0;
    }
    // FFT -> mag^2
    const fftOut = fft(fftIn);
    for (let j = 0; j < fftSize; j++)
      fftOut[j] =
        fftOut[2 * j + 0] * fftOut[2 * j + 0] +
        fftOut[2 * j + 1] * fftOut[2 * j + 1];
    for (let j = 1; j < fftSize / 2; j++) fftOut[j] += fftOut[fftSize - j];
    // mel spectrogram
    for (let j = 0; j < mel.nMel; j++) {
      let sum = 0.0;
      for (let k = 0; k < nFft; k++) sum += fftOut[k] * filters[j * nFft + k];
      if (sum < 1e-10) sum = 1e-10;
      sum = Math.log10(sum);
      mel.data[j * mel.nLen + i] = sum;
    }
  }

  // clamping and normalization
  let mmax = -1e20;
  for (let i = 0; i < mel.nMel * mel.nLen; i++)
    if (mel.data[i] > mmax) mmax = mel.data[i];
  mmax -= 8.0;

  for (let i = 0; i < mel.nMel * mel.nLen; i++) {
    if (mel.data[i] < mmax) mel.data[i] = mmax;
    mel.data[i] = (mel.data[i] + 4.0) / 4.0;
  }

  return mel;
};

const byteDecoder = (() => {
  const range = (from, to) =>
    Array.from({ length: to - from + 1 }, (_, i) => i + from);
  const ps = ["!~", "¡¬", "®ÿ"]
    .map((v) => v.split("").map((v) => v.charCodeAt(0)))
    .flatMap(([k, v]) => range(k, v + 1));
  const n3 = range(0, 2 ** 8).filter((v) => !ps.includes(v));
  const bs = [...ps, ...n3];
  const cs = [...ps, ...n3.map((_, i) => 2 ** 8 + i)];
  return Object.freeze(
    Object.fromEntries(cs.map((n, i) => [String.fromCharCode(n), bs[i]]))
  );
})();

export const decode = (data, vocab) =>
  new TextDecoder().decode(
    new Uint8Array(
      Array.from(data, (x) => Number(x))
        .map((token) => vocab[token])
        .join("")
        .split("")
        .map((str) => byteDecoder[str])
    )
  );

const reshape = (arr: number[], dim: number[]): any[] => {
  let elemIndex = 0;

  if (!dim || !arr) return [];

  function _nest(dimIndex: number): any[] {
    let result: any[] = [];

    if (dimIndex === dim.length - 1) {
      result = result.concat(arr.slice(elemIndex, elemIndex + dim[dimIndex]));
      elemIndex += dim[dimIndex];
    } else {
      for (let i = 0; i < dim[dimIndex]; i++) {
        result.push(_nest(dimIndex + 1));
      }
    }

    return result;
  }
  return _nest(0);
};

export const trimOrPad = (mel: Mel, max: number): Mel => ({
  nMel: mel.nMel,
  nLen: max,
  data: new Float32Array(
    reshape([...mel.data], [mel.nMel, mel.nLen])
      .flatMap((v) => v.concat(Array(max).fill(0)).slice(0, max))
      .flat()
  ),
});

type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export const concatTypedArray = (...arr: TypedArray[]): TypedArray => {
  const newArray = new (arr[0].constructor.bind.apply(arr[0].constructor, [
    null,
    arr.reduce((acc, buffer) => acc + buffer.length, 0),
  ]))();
  let offset = 0;
  for (let typedArr of arr) {
    newArray.set(typedArr, offset);
    offset += typedArr.length;
  }
  return newArray;
};

export const resample = (
  array: TypedArray,
  sampleRateOld,
  sampleRateNew
): TypedArray => {
  if (sampleRateNew === sampleRateOld) return array;
  const factor = sampleRateNew / sampleRateOld;
  const newLength = Math.round(array.length * factor);
  const result = new (array.constructor.bind.apply(array.constructor, [
    null,
    newLength,
  ]))();
  for (let i = 0; i < newLength; i++) result[i] = array[Math.floor(i / factor)];
  return result;
};
