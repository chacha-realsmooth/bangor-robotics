//                ASR (Whistle)                 //
// Speech to text with preference towards the   //
// words in tools.ts.                           //

import { createWhistle } from "cactus-needle/browser";
import { keywords } from "./tools";

declare const __NEEDLE_VERSION__: string;

// Whistle accepts at most 30 seconds of 16 kHz audio per call
const MAX_SAMPLES = 30 * 16_000;

export type Asr = {
  // 16 kHz mono samples in, trimmed text out ("" for silence)
  transcribe(samples: Float32Array): Promise<string>;
  close(): Promise<void>;
};

// Whistle rejects samples outside [-1, 1], and browser audio can overshoot slightly
function clampSamples(samples: Float32Array): void {
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];

    if (sample > 1) {
      samples[i] = 1;
    } else if (sample < -1) {
      samples[i] = -1;
    } else if (Number.isNaN(sample)) {
      samples[i] = 0;
    }
  }
}

export async function loadAsr(onProgress: (percent: number) => void): Promise<Asr> {
  const whistle = await createWhistle({
    wasmUrl: `https://cdn.jsdelivr.net/npm/cactus-needle@${__NEEDLE_VERSION__}/vendor/needle.wasm`,
    onDownloadProgress: ({ percentage }) => {
      if (percentage != null) onProgress(percentage);
    }
  });

  const vocabulary = keywords();

  async function transcribe(samples: Float32Array): Promise<string> {
    if (samples.length > MAX_SAMPLES) {
      samples = samples.subarray(-MAX_SAMPLES); // keep the most recent 30 s
    }
    clampSamples(samples);

    // The VAD already gives 16 kHz mono, so the raw samples go straight in with no resampling
    const result = await whistle.transcribe(samples, { language: "en", keywords: vocabulary });
    return result.text.trim();
  }

  return {
    transcribe,
    close: () => whistle.close()
  };
}
