//                Embed (Needle)                //
// Ranks commands by similarity to examples,    //
// and interprets wording the parser misses.    //
// One session does both, on a single worker.   //

import { createNeedle, DEFAULT_MODEL, tool, type Tool } from "cactus-needle/browser";
import { SYSTEM, TOOLS, jsonSchema, type Args, type Tools } from "./tools";

declare const __NEEDLE_VERSION__: string;

export type Call = { command: string; args: Args };

export type Prediction = {
  calls: Call[];
  suppressed: Call[]; // calls Needle held back as ungrounded or negated
  confidence: number | null;
  reasoning: string;
};

export type Embed = {
  // How close the text is to each command's examples, roughly -1 to 1
  rank(text: string): Promise<Record<string, number>>;
  predict(text: string): Promise<Prediction>;
  close(): Promise<void>;
};

type Vectors = Record<string, Float32Array[]>;

// ---------- Ranking ----------

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function averageOf(vectors: Float32Array[]): Float32Array {
  const average = new Float32Array(vectors[0]?.length ?? 0);
  for (const vector of vectors) {
    for (let i = 0; i < vector.length; i++) {
      average[i] += vector[i] / vectors.length;
    }
  }
  return average;
}

// Raw Needle embeddings all score about 0.9 against each other, which tells
// commands apart badly. Subtracting the average of the examples first, then
// scaling to length 1, spreads the scores out.
export function createRanker(examples: Vectors): (vector: Float32Array) => Record<string, number> {
  const average = averageOf(Object.values(examples).flat());

  function centre(vector: Float32Array): Float32Array {
    const centred = new Float32Array(vector.length);
    let length = 0;

    for (let i = 0; i < vector.length; i++) {
      centred[i] = vector[i] - average[i];
      length += centred[i] * centred[i];
    }

    length = Math.sqrt(length) || 1;
    for (let i = 0; i < vector.length; i++) {
      centred[i] /= length;
    }
    return centred;
  }

  const centredExamples: Vectors = {};
  for (const [command, vectors] of Object.entries(examples)) {
    centredExamples[command] = vectors.map(centre);
  }

  // A command's score is how close the text is to its closest example
  return vector => {
    const text = centre(vector);
    const scores: Record<string, number> = {};

    for (const [command, vectors] of Object.entries(centredExamples)) {
      let best = vectors.length > 0 ? -Infinity : 0;
      for (const example of vectors) {
        best = Math.max(best, dot(example, text));
      }
      scores[command] = best;
    }
    return scores;
  };
}

// ---------- Example embeddings, cached between visits ----------

const CACHE_NAME = "example-embeddings";

// Changes whenever the examples or the model change, so a stale cache is never used
async function cacheKey(texts: string[]): Promise<string> {
  const source = DEFAULT_MODEL.sha256 + "\n" + texts.join("\n");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));

  let hex = "";
  for (const byte of new Uint8Array(digest)) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return "/example-embeddings/" + hex;
}

// The cache can be missing (e.g. not https) or full. Then we just embed every visit.
async function readCache(key: string): Promise<Float32Array | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(key);
    if (!response) return null;
    return new Float32Array(await response.arrayBuffer());
  } catch {
    return null;
  }
}

async function writeCache(key: string, data: Float32Array): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    for (const old of await cache.keys()) {
      await cache.delete(old);
    }
    await cache.put(key, new Response(data.buffer as ArrayBuffer));
  } catch {
    // Not cached this time, nothing else depends on it
  }
}

// Every example's vector is stored as one long array, one after another
function joinVectors(vectors: Float32Array[]): Float32Array {
  let total = 0;
  for (const vector of vectors) total += vector.length;

  const joined = new Float32Array(total);
  let offset = 0;
  for (const vector of vectors) {
    joined.set(vector, offset);
    offset += vector.length;
  }
  return joined;
}

async function embedExamples(embed: (text: string) => Promise<Float32Array>, tools: Tools): Promise<Vectors> {
  const texts = Object.values(tools).flatMap(spec => spec.examples);
  const key = await cacheKey(texts);

  let joined = await readCache(key);
  if (!joined) {
    const vectors: Float32Array[] = [];
    for (const text of texts) {
      vectors.push(await embed(text));
    }
    joined = joinVectors(vectors);
    await writeCache(key, joined);
  }

  // Split the long array back up by command
  const size = joined.length / texts.length;
  const result: Vectors = {};
  let index = 0;

  for (const [command, spec] of Object.entries(tools)) {
    result[command] = [];
    for (let i = 0; i < spec.examples.length; i++) {
      result[command].push(joined.subarray(index * size, (index + 1) * size));
      index++;
    }
  }
  return result;
}

// ---------- Loading ----------

function needleTools(tools: Tools): Record<string, Tool> {
  const result: Record<string, Tool> = {};
  for (const [name, spec] of Object.entries(tools)) {
    result[name] = tool({
      description: spec.description,
      inputSchema: jsonSchema(spec),
      // Keywords double as Needle's routing hints. With many commands they help it a lot.
      triggers: spec.keywords
    });
  }
  return result;
}

function toCall(call: { toolName: string; input: unknown }): Call {
  return {
    command: call.toolName,
    args: { ...(call.input as Args) }
  };
}

export async function loadEmbed(onProgress: (percent: number) => void, tools: Tools = TOOLS): Promise<Embed> {
  const needle = await createNeedle({
    wasmUrl: `https://cdn.jsdelivr.net/npm/cactus-needle@${__NEEDLE_VERSION__}/vendor/needle.wasm`,
    tools: needleTools(tools),
    system: SYSTEM,
    stateless: true,
    // Enough for Needle's reasoning plus one call. Lower limits changed its
    // answers in testing. This stops a runaway answer after about 1.5 s.
    maxOutputTokens: 80,
    onDownloadProgress: ({ percentage }) => {
      if (percentage != null) onProgress(percentage);
    }
  });

  let score: (vector: Float32Array) => Record<string, number>;
  try {
    const examples = await embedExamples(text => needle.embed(text), tools);
    score = createRanker(examples);
  } catch (err) {
    await needle.close();
    throw err;
  }

  async function rank(text: string): Promise<Record<string, number>> {
    return score(await needle.embed(text));
  }

  async function predict(text: string): Promise<Prediction> {
    const result = await needle.generate({ prompt: text });
    return {
      calls: result.toolCalls.map(toCall),
      suppressed: result.suppressedToolCalls.map(toCall),
      confidence: result.confidence,
      reasoning: result.reasoning
    };
  }

  return {
    rank,
    predict,
    close: () => needle.close()
  };
}
