//                Voice-instruct                //
// The pipeline. VAD > ASR > Parser > send      //
// Subscribe with onChange() and onLevel()      //
// Mic stays on work runs one task at a time    //

import { loadAsr, type Asr } from "./asr";
import { loadEmbed, type Embed, type Prediction } from "./embed";
import { parse } from "./parser";
import { TOOLS, validate, type Args, type Command } from "./tools";
import { CONTINUE_WINDOW_MS, loadVad, type Vad, type VadHandlers } from "./vad";

export const SERVER_URL = "https://robo-dog/api/speech";
const SEND_TIMEOUT_MS = 5_000;

// Needle's pick is selected automatically if it's at least this confident
const MIN_CONFIDENCE = 0.3;

export type Option = {
  command: string;
  args: Args;
  source: "parser" | "needle" | "suppressed" | null; // null = nothing suggested this command
  confidence: number | null;
  similarity: number | null; // only filled in when Needle was used
};

export type State = {
  status: string;
  error: string;
  ready: boolean;
  busy: boolean;
  listening: boolean;
  transcript: string;
  timings: string;
  reasoning: string;
  options: Option[];
  revision: number; // used for UI re-rendering
  selected: string | null;
  draft: Command | null; // the selected command with its current (maybe invalid) arguments
  command: Command | null; // the draft once it's valid (only this can be sent)
};

export type Models = {
  loadVad(handlers: VadHandlers): Promise<Vad>;
  loadAsr(onProgress: (percent: number) => void): Promise<Asr>;
  loadEmbed(onProgress: (percent: number) => void): Promise<Embed>;
  fetch: typeof fetch;
};

const defaultModels: Models = {
  loadVad,
  loadAsr,
  loadEmbed,
  fetch: (...args) => fetch(...args)
};

type Utterance = { stale: boolean };

function msSince(start: number): string {
  return Math.round(performance.now() - start) + " ms";
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function optionsFromParser(parsed: Command): Option[] {
  const options: Option[] = [];

  for (const command of Object.keys(TOOLS)) {
    if (command === parsed.command) {
      options.push({ command, args: parsed.args, source: "parser", confidence: null, similarity: null });
    } else {
      options.push({ command, args: {}, source: null, confidence: null, similarity: null });
    }
  }
  return options;
}

function optionsFromNeedle(prediction: Prediction, similarity: Record<string, number>): Option[] {
  const options: Option[] = [];

  for (const command of Object.keys(TOOLS)) {
    const call = prediction.calls.find(c => c.command === command);
    const withheld = prediction.suppressed.find(c => c.command === command);

    let source: Option["source"] = null;
    let args: Args = {};
    if (call) {
      source = "needle";
      args = { ...call.args };
    } else if (withheld) {
      source = "suppressed";
      args = { ...withheld.args };
    }

    options.push({
      command,
      args,
      source,
      confidence: call ? prediction.confidence : null,
      similarity: similarity[command] ?? null
    });
  }
  return options;
}

// Needle's single pick, if its confident or the similarity ranking agrees with it
function needlePick(prediction: Prediction, similarity: Record<string, number>): string | null {
  if (prediction.calls.length !== 1) return null;
  const pick = prediction.calls[0].command;

  const confident = prediction.confidence === null || prediction.confidence >= MIN_CONFIDENCE;
  if (confident) return pick;

  let closest: string | null = null;
  for (const [command, score] of Object.entries(similarity)) {
    if (closest === null || score > similarity[closest]) closest = command;
  }
  return pick === closest ? pick : null;
}

export class VoiceInstruct {
  state: State = {
    status: "Models not loaded",
    error: "",
    ready: false,
    busy: false,
    listening: false,
    transcript: "",
    timings: "",
    reasoning: "",
    options: [],
    revision: 0,
    selected: null,
    draft: null,
    command: null
  };

  private changeHandlers = new Set<(state: State) => void>();
  private levelHandlers = new Set<(probability: number) => void>();
  private models: Models;

  private vad: Vad | null = null;
  private asr: Asr | null = null;
  private embed: Embed | null = null;

  private text = ""; // what was said or typed for the command under review
  private queue: Promise<void> = Promise.resolve();
  private queued = 0;
  private latest: Utterance | null = null;
  private speaking = false;

  constructor(models: Models = defaultModels) {
    this.models = models;
  }

  // Calls the handler after every change to the state. Returns a function that removes it
  onChange(handler: (state: State) => void): () => void {
    this.changeHandlers.add(handler);
    return () => this.changeHandlers.delete(handler);
  }

  // Calls the handler with the microphone level, from 0 to 1. Returns a function that removes it
  onLevel(handler: (probability: number) => void): () => void {
    this.levelHandlers.add(handler);
    return () => this.levelHandlers.delete(handler);
  }

  private set(changes: Partial<State>): void {
    this.state = { ...this.state, ...changes };
    for (const handler of this.changeHandlers) handler(this.state);
  }

  private setLevel(probability: number): void {
    for (const handler of this.levelHandlers) handler(probability);
  }

  // ---------- Loading ----------

  // Silero, Whistle and Needle each run in their own worker, so they load at the same time
  async load(): Promise<void> {
    if (this.state.busy || this.state.ready) return;

    const progress = { Whistle: 0, Needle: 0 };
    const report = (name: "Whistle" | "Needle") => (percent: number) => {
      progress[name] = Math.round(percent);
      this.set({ status: `Loading models: Whistle ${progress.Whistle}%, Needle ${progress.Needle}%` });
    };

    this.set({ busy: true, error: "", status: "Loading models..." });
    const started = performance.now();

    const [vad, asr, embed] = await Promise.allSettled([
      this.models.loadVad(this.vadHandlers()),
      this.models.loadAsr(report("Whistle")),
      this.models.loadEmbed(report("Needle"))
    ]);

    if (vad.status === "fulfilled" && asr.status === "fulfilled" && embed.status === "fulfilled") {
      this.vad = vad.value;
      this.asr = asr.value;
      this.embed = embed.value;
      this.set({ busy: false, ready: true, status: "Ready", timings: "Models loaded in " + msSince(started) });
      return;
    }

    // Close whatever did load, so a retry starts clean
    if (vad.status === "fulfilled") await vad.value.destroy().catch(() => {});
    if (asr.status === "fulfilled") await asr.value.close().catch(() => {});
    if (embed.status === "fulfilled") await embed.value.close().catch(() => {});

    const failed = [vad, asr, embed].find(result => result.status === "rejected") as PromiseRejectedResult;
    this.set({ busy: false, status: "Loading failed", error: errorMessage(failed.reason) });
  }

  // ---------- Listening ----------

  private vadHandlers(): VadHandlers {
    const idle = () => this.state.listening && !this.state.busy;

    return {
      onSpeechStart: () => {
        this.speaking = true;
        if (idle()) this.set({ status: "Hearing speech..." });
      },
      onMisfire: () => {
        this.speaking = false;
        if (idle()) this.set({ status: "Listening..." });
      },
      onSpeechEnd: (samples, continues) => {
        this.speaking = false;
        if (this.state.listening) this.hear(samples, continues);
      },
      onLevel: probability => {
        this.setLevel(this.state.listening ? probability : 0);
      }
    };
  }

  async toggleListening(): Promise<void> {
    if (!this.vad) return;

    if (this.state.listening) {
      this.set({ listening: false, status: "Microphone off" });
      await this.vad.stop().catch(err => this.set({ error: errorMessage(err) }));
      return;
    }

    this.set({ listening: true, error: "", status: "Listening..." });
    try {
      await this.vad.start();
    } catch (err) {
      this.set({ listening: false, status: "Microphone unavailable", error: errorMessage(err) });
      await this.vad.stop().catch(() => {});
    }
  }

  // ---------- Work queue ----------

  // Runs tasks one at a time, in the order they were added, and shows any error
  private run(task: () => Promise<void>): Promise<void> {
    this.queued++;
    this.set({ busy: true });

    this.queue = this.queue.then(async () => {
      try {
        this.set({ error: "" });
        await task();
      } catch (err) {
        this.set({ status: "Error", error: errorMessage(err) });
      } finally {
        this.queued--;
        if (this.queued === 0) this.set({ busy: false });
      }
    });
    return this.queue;
  }

  private clear(): void {
    this.text = "";
    this.set({
      options: [],
      revision: this.state.revision + 1,
      reasoning: "",
      selected: null,
      draft: null,
      command: null
    });
  }

  // ---------- Understanding speech and text ----------

  // If the speaker carries on, the VAD sends the whole thing again with
  // `continues` set, and the earlier utterance is marked stale. Whistle and
  // Needle can't be stopped mid-run, so stale work is dropped at its next step.
  private hear(samples: Float32Array, continues: boolean): void {
    if (continues && this.latest) this.latest.stale = true;

    const utterance: Utterance = { stale: false };
    this.latest = utterance;
    const heard = performance.now();

    void this.run(async () => {
      if (utterance.stale) return;

      this.set({ status: "Transcribing..." });
      const started = performance.now();
      const text = await this.asr!.transcribe(samples);
      if (utterance.stale) return;

      const asrTime = "Whistle " + msSince(started);
      if (!text) {
        // Probably just noise, so leave any command under review alone
        this.set({ status: "No speech recognised", timings: asrTime });
        return;
      }

      this.clear();
      this.set({ transcript: text });

      const step = await this.understand(text, () => this.carriedOn(utterance, heard));
      if (step) this.set({ timings: `${asrTime} | ${step} | Total ${msSince(heard)}` });
    });
  }

  // Waits until the speaker can no longer carry on, then says whether they did
  // Needle takes about a second, so it only starts once this returns false
  private async carriedOn(utterance: Utterance, heard: number): Promise<boolean> {
    const wait = heard + CONTINUE_WINDOW_MS - performance.now();

    if (wait > 0) {
      await sleep(wait);
      while (this.speaking && !utterance.stale) {
        await sleep(50);
      }
    }
    return utterance.stale;
  }

  // Interprets typed text as if it had been spoken
  async interpret(text: string): Promise<void> {
    text = text.trim();
    if (!text || !this.state.ready || this.state.busy) return;

    await this.run(async () => {
      this.clear();
      this.set({ transcript: text });
      const step = await this.understand(text);
      this.set({ timings: step ?? "" });
    });
  }

  // Fills in the command options. Returns a timing label, or null if the speaker carried on meanwhile.
  private async understand(text: string, carriedOn = async () => false): Promise<string | null> {
    const started = performance.now();
    const parsed = parse(text);

    let options: Option[];
    let reasoning: string;
    let pick: string | null;
    let step: string;

    if (parsed) {
      options = optionsFromParser(parsed);
      reasoning = "Exact phrase match, Needle skipped. Steps count as seconds.";
      pick = parsed.command;
      step = "Parser";
    } else {
      if (await carriedOn()) return null;
      this.set({ status: "Needle interpreting..." });

      const [prediction, similarity] = await Promise.all([
        this.embed!.predict(text),
        this.embed!.rank(text)
      ]);
      if (await carriedOn()) return null;

      options = optionsFromNeedle(prediction, similarity);
      reasoning = prediction.reasoning;
      pick = needlePick(prediction, similarity);
      step = "Needle";
    }

    this.text = text;
    this.set({
      options,
      reasoning,
      revision: this.state.revision + 1,
      status: "Review the command, then send"
    });
    if (pick) this.choose(pick);

    return step + " " + msSince(started);
  }

  // ---------- Reviewing and sending ----------

  select(command: string): void {
    if (!this.state.busy) this.choose(command);
  }

  private choose(command: string): void {
    const option = this.state.options.find(o => o.command === command);
    if (!option) return;

    const draft = { command, args: { ...option.args } };
    this.set({ selected: command, draft, command: validate(draft) });
  }

  // Sets one argument of the selected command. null or "" removes it
  setArg(name: string, value: string | number | null): void {
    if (!this.state.draft || this.state.busy) return;

    const args = { ...this.state.draft.args };
    if (value === null || value === "" || Number.isNaN(value)) {
      delete args[name];
    } else {
      args[name] = value;
    }

    const draft = { command: this.state.draft.command, args };
    this.set({ draft, command: validate(draft) });
  }

  async send(): Promise<void> {
    const command = this.state.command;
    if (!command || this.state.busy) return;

    await this.run(async () => {
      this.set({ status: "Sending..." });

      const response = await this.models.fetch(SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: this.text, command: command.command, args: command.args }),
        signal: AbortSignal.timeout(SEND_TIMEOUT_MS)
      });
      if (!response.ok) throw new Error("Server returned HTTP " + response.status);

      this.clear();
      this.set({ status: "Command sent" });
    });
  }

  async dispose(): Promise<void> {
    this.set({ listening: false });
    this.changeHandlers.clear();
    this.levelHandlers.clear();
    await Promise.allSettled([this.vad?.destroy(), this.asr?.close(), this.embed?.close()]);
  }
}
