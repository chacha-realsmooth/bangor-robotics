//                    Tools                     //
// Every robot command lives here. The parser,  //
// Needle's schemas, validation, examples and   //
// ASR keywords are all built from this file.   //

export type EnumParam = {
  type: "enum";
  description: string;
  // Value sent to the robot and synonyms the parser accepts for it
  values: Record<string, readonly string[]>;
};

export type NumberParam = {
  type: "number";
  description: string;
  min: number;
  max: number;
  // Words the parser accepts after the number e.g. "seconds"
  units: readonly string[];
};

export type Param = EnumParam | NumberParam;

export type ToolSpec = {
  description: string;

  // Every parameter is required
  params: Record<string, Param>;

  // Whole sentences the parser accepts:
  //   word      that exact word
  //   (a|b c)   one of these
  //   [a|b c]   optionally one of these
  //   {param}   a parameter (a number must be followed by one of its units)
  phrases: readonly string[];

  // Everyday wordings, used to rank commands by similarity
  examples: readonly string[];

  // Words Whistle should listen out for. Parameter words are added automatically
  keywords: readonly string[];
};

export type Tools = Record<string, ToolSpec>;
export type Args = Record<string, string | number>;
export type Command = { command: string; args: Args };

// System Prompt given to Needle with every request
export const SYSTEM = "Map robot dog commands to tools. Left and right are turns. Never invent missing parameters.";

// Parameters shared by several commands

const seconds: NumberParam = {
  type: "number",
  description: "Seconds.",
  min: 0.1,
  max: 3,
  units: ["seconds", "second", "secs", "sec", "s", "steps", "step"]
};

const leg: EnumParam = {
  type: "enum",
  description: "Which leg.",
  values: {
    front_left: ["front left", "left front", "fore left", "left fore"],
    front_right: ["front right", "right front", "fore right", "right fore"],
    back_left: ["back left", "left back", "rear left", "left rear", "hind left", "left hind", "behind left"],
    back_right: ["back right", "right back", "rear right", "right rear", "hind right", "right hind", "behind right"]
  }
};

export const TOOLS: Tools = {
  move: {
    description: "Walk forward or backward for a duration in seconds. Steps equal seconds. Never invent the duration.",
    params: {
      direction: {
        type: "enum",
        description: "Direction to walk.",
        values: {
          forward: ["forwards", "ahead", "straight", "straight ahead", "onward", "onwards"],
          backward: ["backwards", "back", "back up", "reverse", "in reverse"]
        }
      },
      duration: seconds
    },
    phrases: [
      "[move|moved|go|walk|run|head|step|trot|back] {direction} [for] {duration}",
      "(walk|go|move) {duration} {direction}"
    ],
    examples: [
      "move forward for two seconds",
      "walk backwards for one second",
      "go straight ahead three steps",
      "back up a little",
      "take a few steps forward"
    ],
    keywords: ["move", "walk", "step", "steps"]
  },
  turn: {
    description: "Turn on the spot to the left or right for a duration in seconds. Never invent the duration.",
    params: {
      direction: {
        type: "enum",
        description: "Direction to turn.",
        values: {
          left: ["anticlockwise", "counterclockwise", "counter clockwise"],
          right: ["clockwise"]
        }
      },
      duration: seconds
    },
    phrases: [
      "[turn|turned|rotate|spin|move|go|walk] [to the] {direction} [for] {duration}",
      "(turn|rotate|spin) {duration} [to the] {direction}"
    ],
    examples: [
      "turn left for one second",
      "rotate right for two seconds",
      "spin clockwise for half a second",
      "go left three steps",
      "face the other way"
    ],
    keywords: ["turn", "rotate", "spin"]
  },
  lift_leg: {
    description: "Lift one leg off the ground.",
    params: { leg },
    phrases: [
      "(lift|raise|lifts|raised|lifted) [up] [your|the|its] {leg} [leg|paw|foot]",
      "give [me] [your|the] {leg} [leg|paw|foot]"
    ],
    examples: [
      "lift your front left leg",
      "raise the back right paw",
      "give me your paw",
      "pick up your rear left foot"
    ],
    keywords: ["lift", "raise", "leg", "paw", "foot"]
  },
  lower_leg: {
    description: "Put a lifted leg back down.",
    params: { leg },
    phrases: [
      "(lower|drop) [your|the|its] {leg} [leg|paw|foot]",
      "(put|but) [your|the|its] {leg} [leg|paw|foot] down",
      "put down [your|the|its] {leg} [leg|paw|foot]"
    ],
    examples: [
      "lower your front left leg",
      "put the back right paw down",
      "drop your rear left foot",
      "put your leg back on the ground"
    ],
    keywords: ["lower", "drop", "put down", "leg", "paw", "foot"]
  },
  sit: {
    description: "Sit down.",
    params: {},
    phrases: ["sit [down]", "take a seat"],
    examples: ["sit", "sit down", "take a seat", "have a seat"],
    keywords: ["sit", "sit down", "seat"]
  },
  stand: {
    description: "Stand up.",
    params: {},
    phrases: ["stand [up]", "get up", "on your feet"],
    examples: ["stand up", "get up", "stand", "on your feet"],
    keywords: ["stand", "stand up"]
  },
  lie_down: {
    description: "Lie down flat.",
    params: {},
    phrases: ["(lie|lay) [down|flat]", "get down"],
    examples: ["lie down", "lay down", "lie flat", "get down on the floor"],
    keywords: ["lie down", "lay down"]
  },
  stop: {
    description: "Stop all movement immediately.",
    params: {},
    phrases: [
      "stop [moving|the robot|now|right now|everything|it]",
      "(halt|freeze|abort|emergency stop|stay still|keep still|hold still|stand still)",
      "(don't|do not) move"
    ],
    examples: [
      "stop",
      "halt",
      "stop moving",
      "freeze",
      "stay still",
      "emergency stop"
    ],
    keywords: ["stop", "halt", "freeze", "abort", "emergency stop", "still"]
  },
  status: {
    description: "Get robot status.",
    params: {},
    phrases: [
      "[get|check|show|report] [the] [robot|your|robot's] status",
      "status report",
      "(what's|what is) [your|the robot's] status",
      "how are you [doing]"
    ],
    examples: [
      "robot status",
      "how are you doing",
      "report your battery level",
      "are you ok",
      "what is your state"
    ],
    keywords: ["status", "report"]
  }
};

// The JSON Schema Needle expects for a command's parameters
export function jsonSchema(spec: ToolSpec) {
  const properties: Record<string, { type: string; description: string; enum?: string[] }> = {};

  for (const [name, param] of Object.entries(spec.params)) {
    if (param.type === "enum") {
      properties[name] = { type: "string", enum: Object.keys(param.values), description: param.description };
    } else {
      properties[name] = { type: "number", description: param.description };
    }
  }

  return { type: "object" as const, properties, required: Object.keys(spec.params) };
}

function isValidValue(param: Param, value: unknown): boolean {
  if (param.type === "enum") {
    return typeof value === "string" && Object.hasOwn(param.values, value);
  }
  return typeof value === "number" && Number.isFinite(value) && value >= param.min && value <= param.max;
}

// Returns a clean copy if it names a command and every parameter is there and
// in range, otherwise null. Only a validated command can be sent to the robot
export function validate(candidate: unknown, tools: Tools = TOOLS): Command | null {
  if (!candidate || typeof candidate !== "object") return null;

  const { command, args } = candidate as { command?: unknown; args?: unknown };
  if (typeof command !== "string" || !Object.hasOwn(tools, command)) return null;
  if (!args || typeof args !== "object" || Array.isArray(args)) return null;

  const params = tools[command].params;
  const given = args as Record<string, unknown>;

  for (const key of Object.keys(given)) {
    if (!Object.hasOwn(params, key)) return null;
  }

  const clean: Args = {};
  for (const [name, param] of Object.entries(params)) {
    const value = given[name];
    if (!isValidValue(param, value)) return null;
    clean[name] = value as string | number;
  }
  return { command, args: clean };
}

// Words for Whistle to listen out for (each command's keywords plus its parameter words)
// Single letters and contractions are left out because they made Whistle worse in testing
export function keywords(tools: Tools = TOOLS): string[] {
  const words = new Set<string>();

  for (const spec of Object.values(tools)) {
    for (const word of spec.keywords) words.add(word);

    for (const param of Object.values(spec.params)) {
      if (param.type === "number") {
        for (const unit of param.units) words.add(unit);
      } else {
        for (const [value, synonyms] of Object.entries(param.values)) {
          words.add(value);
          for (const synonym of synonyms) words.add(synonym);
        }
      }
    }
  }

  return [...words].filter(word => /^[a-z]{2,}( [a-z]+)*$/.test(word));
}

// Throws on a broken command definition, so mistakes show up as soon as the app starts
export function assertTools(tools: Tools = TOOLS): void {
  const problem = (message: string) => new Error("tools.ts: " + message);
  const validName = /^[a-z_][a-z0-9_]*$/i;

  for (const [name, spec] of Object.entries(tools)) {
    if (!/^[a-z_][a-z0-9_]*$/.test(name)) throw problem(`invalid tool name "${name}"`);
    if (!spec.description) throw problem(`${name} needs a description`);
    if (spec.phrases.length === 0 && spec.examples.length === 0) throw problem(`${name} needs phrases or examples`);

    for (const [paramName, param] of Object.entries(spec.params)) {
      if (!validName.test(paramName)) throw problem(`${name}.${paramName} is not a valid parameter name`);

      if (param.type === "number") {
        if (!(param.min < param.max)) throw problem(`${name}.${paramName} needs min < max`);
        continue;
      }

      // The same word can't mean two different values
      const meaning = new Map<string, string>();
      for (const [value, synonyms] of Object.entries(param.values)) {
        for (const word of [value, ...synonyms]) {
          if (meaning.has(word)) throw problem(`${name}.${paramName}: "${word}" means both ${meaning.get(word)} and ${value}`);
          meaning.set(word, value);
        }
      }
      if (meaning.size === 0) throw problem(`${name}.${paramName} needs at least one value`);
    }
  }
}

assertTools();
