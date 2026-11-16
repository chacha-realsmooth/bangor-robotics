//                 VAD (Silero)                 //
// Hands over each utterance 100 ms after       //
// speech stops. Speech that resumes within     //
// 600 ms is joined on and replaces it.         //

import type { MicVAD } from "@ricky0123/vad-web";

declare const __VAD_VERSION__: string;
declare const __ORT_VERSION__: string;

const END_SILENCE_MS = 100;
export const CONTINUE_WINDOW_MS = 600;
const LEVEL_INTERVAL_MS = 100;

// A joined utterance stops growing at 20 s (Whistle's limit is 30 s)
const MAX_SAMPLES = 20 * 16_000;

export type VadHandlers = {
  onSpeechStart(): void;
  // `continues` means these samples include, and replace, the previous utterance
  onSpeechEnd(samples: Float32Array, continues: boolean): void;
  onMisfire(): void;
  // Chance that the current audio is speech, 0 to 1, at most every 100 ms
  onLevel(probability: number): void;
};

export type Vad = {
  start(): Promise<void>; 
  stop(): Promise<void>;
  destroy(): Promise<void>;
};

type Utterance = {
  samples: Float32Array;
  endedAt: number;
};

function join(first: Float32Array, second: Float32Array): Float32Array {
  const joined = new Float32Array(first.length + second.length);
  joined.set(first);
  joined.set(second, first.length);
  return joined;
}

export async function loadVad(handlers: VadHandlers): Promise<Vad> {
  let stream: MediaStream | null = null;
  let lastLevelAt = 0;
  let previous: Utterance | null = null;
  let continuing = false;

  async function getStream(): Promise<MediaStream> {
    const live = stream?.getAudioTracks().some(track => track.readyState === "live");
    if (!live) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
    }
    return stream!;
  }

  function release(): void {
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
    }
    stream = null;
    previous = null;
  }

  function onFrameProcessed({ isSpeech }: { isSpeech: number }): void {
    const now = performance.now();
    if (now - lastLevelAt < LEVEL_INTERVAL_MS) return;

    lastLevelAt = now;
    handlers.onLevel(isSpeech);
  }

  function onSpeechStart(): void {
    continuing = previous !== null && performance.now() - previous.endedAt <= CONTINUE_WINDOW_MS;
    handlers.onSpeechStart();
  }

  function onSpeechEnd(samples: Float32Array): void {
    let continues = false;

    if (continuing && previous && previous.samples.length + samples.length <= MAX_SAMPLES) {
      // The new part starts with up to 300 ms of the audio before it, so the join has no real gap
      samples = join(previous.samples, samples);
      continues = true;
    }

    previous = { samples, endedAt: performance.now() };
    handlers.onSpeechEnd(samples, continues);
  }

  const { MicVAD } = await import("@ricky0123/vad-web");
  const vad: MicVAD = await MicVAD.new({
    model: "v6",
    startOnLoad: false,
    baseAssetPath: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${__VAD_VERSION__}/dist/`,
    onnxWASMBasePath: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${__ORT_VERSION__}/dist/`,
    ortConfig(ort) {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
    },

    getStream,
    pauseStream: async () => {},
    resumeStream: getStream,

    redemptionMs: END_SILENCE_MS,
    preSpeechPadMs: 300,
    // Short enough to keep a quick follow-on like "two", long enough to ignore clicks
    minSpeechMs: 160,
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,

    onFrameProcessed,
    onSpeechStart,
    onSpeechEnd,
    onVADMisfire: () => handlers.onMisfire()
  });

  async function start(): Promise<void> {
    await vad.start();
    if (vad.errored) throw new Error(vad.errored);
  }

  async function stop(): Promise<void> {
    try {
      await vad.pause();
      handlers.onLevel(0);
    } finally {
      release();
    }
  }

  async function destroy(): Promise<void> {
    release();
    await vad.destroy();
  }

  return { start, stop, destroy };
}
