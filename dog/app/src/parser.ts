//                  Fast-parser                 //
// Turns phrase templates in tools.ts into      //
// regex. Anything that does not noisily match  //
// a phrase returns null and goes through embed //

import { TOOLS, type Args, type Command, type Param, type Tools } from "./tools";

const UNITS: Record<string, number | undefined> = {
  zero: 0,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  thirteen: 13,
  fourteen: 14,
  fifteen: 15,
  sixteen: 16,
  seventeen: 17,
  eighteen: 18,
  nineteen: 19,
};

const TENS: Record<string, number | undefined> = {
  twenty: 20,
  thirty: 30,
  forty: 40,
  fifty: 50,
  sixty: 60,
  seventy: 70,
  eighty: 80,
  ninety: 90,
};

const DENOMINATORS: Record<string, number | undefined> = {
  half: 2,
  halves: 2,
  third: 3,
  thirds: 3,
  quarter: 4,
  quarters: 4,
};

const INVALID_WORDS = new Set([
  "plus", "minus", "times", "divided", "multiply", "multiplied",
  "by", "over", "or", "instead", "rather", "than",
  "to", "million", "billion", "squared", "cubed",
]);

function hasInvalidExpression(text: string): boolean {
  if (/[+\-*/=]/.test(text)) return true;
  return text.split(" ").some(word => INVALID_WORDS.has(word));
}

// Intentionally permissive to combat ASR errors
// "one two" returns 1, rather than null
function wholeNumber(words: string[]): number | null {
  let total = 0, group = 0;
  let prev: "start" | "a" | "unit" | "tens" | "scale" = "start";
  let usedHundred = false;
  let usedThousand = false;

  for (const word of words) {
    const unit = UNITS[word], tens = TENS[word];

    if (word === "and") {
      continue;
    } else if (word === "a" || word === "an") {
      if (prev === "start") {
        group = 1;
        prev = "a";
      }
    } else if (unit !== undefined) {
      if (prev === "start" || prev === "scale" ||
          (prev === "tens" && unit > 0 && unit < 10)) {
        group += unit;
        prev = "unit";
      }
    } else if (tens !== undefined) {
      if (prev === "start" || prev === "scale") {
        group += tens;
        prev = "tens";
      }
    } else if (word === "hundred") {
      if (!usedHundred && group > 0 && group < 100) {
        group *= 100;
        usedHundred = true;
        prev = "scale";
      }
    } else if (word === "thousand") {
      if (!usedThousand && group > 0) {
        total += group * 1000;
        group = 0;
        usedThousand = true;
        usedHundred = false;
        prev = "scale";
      }
    }
  }

  return prev === "start" ? null : total + group;
}

// Plain digits with an optional decimal point: "2", "0.5" or ".5"
function isWrittenInDigits(text: string): boolean {
  return /^\d*\.?\d+$/.test(text);
}

function lastWord(words: string[]): string {
  return words[words.length - 1] ?? "";
}

export function parseNumber(text: string): number | null {
  if (hasInvalidExpression(text)) return null;

  if (isWrittenInDigits(text)) return Number(text);

  const words = text.split(" ");

  // "half a second" and "three quarters of a second" arrive here as
  // "half a" and "three quarters of a", so drop the "a" or "of a" on the end.
  // That's only allowed after a fraction word: "two of a" isn't a number.
  if (words.length > 1 && (lastWord(words) === "a" || lastWord(words) === "an")) {
    words.pop();
    if (lastWord(words) === "of") words.pop();
    if (!DENOMINATORS[lastWord(words)]) return null;
  }

  const point = words.indexOf("point");
  if (point >= 0) {
    const whole = point === 0 ? 0 : wholeNumber(words.slice(0, point));
    const digits = words.slice(point + 1).map(word => UNITS[word]);
    if (whole === null || !digits.length || digits.some(d => d === undefined || d > 9)) return null;
    return Number(whole + "." + digits.join(""));
  }

  const denominator = DENOMINATORS[words[words.length - 1]];
  if (denominator) {
    const and = words.lastIndexOf("and");
    const whole = and === -1 ? 0 : wholeNumber(words.slice(0, and));
    const top = words.slice(and + 1, -1);
    const numerator = top.length ? wholeNumber(top) : 1;
    return whole === null || numerator === null ? null : whole + numerator / denominator;
  }

  return wholeNumber(words);
}

export function normalize(text: string): string {
  let result = text.normalize("NFKC").toLowerCase();      // Unicode normalization, lowercase

  result = result.replace(/[‘’]/g, "'");                  // curly apostrophes
  result = result.replace(/(\p{L})-(?=\p{L})/gu, "$1 ");  // "back-right" -> "back right"
  result = result.replace(/,/g, "");                      // commas ("1,000" -> "1000")
  result = result.replace(/[.!]+(?=\s|$)/g, " ");         // . and ! at end
  result = result.replace(/\s+/g, " ").trim();            // collapse whitespace

  result = result.replace(/^please /, "");                // "please" at the start
  result = result.replace(/ please$/, "");                // "please" at the end
  return result;
}

type Slot = { name: string; param: Param };

type Rule = {
  command: string;
  regex: RegExp;
  slots: Slot[];
};

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");   // Escape regex special characters
}

function alternatives(words: readonly string[]): string {
  const sorted = [...words].sort((a, b) => b.length - a.length); // Longest first, so "straight ahead" is tried before "straight"
  return sorted.map(escapeRegex).join("|");
}

function slotPattern(name: string, param: Param): string {
  if (param.type === "enum") {
    const words: string[] = [];
    for (const [value, synonyms] of Object.entries(param.values)) {
      words.push(value, ...synonyms);
    }
    return `(?<${name}>${alternatives(words)}) `;
  }

  if (param.units.length === 0) {
    return `(?<${name}>\\S+) `;
  }
  return `(?<${name}>.+?) (?:${alternatives(param.units)}) `;
}

// Every piece of the pattern ends with a space, and so does the input
// so an optional piece can just be wrapped in (?: ... )?
function compile(command: string, rawPhrase: string, params: Record<string, Param>): Rule {
  const phrase = rawPhrase.toLowerCase();
  const problem = (message: string) => new Error(`tools.ts: ${command} phrase "${rawPhrase}" ${message}`);

  if (/\S[{[(]|[})\]]\S/.test(phrase)) {
    throw problem("has a slot or bracket touching other characters");
  }

  const group = (inside: string) => alternatives(inside.split("|"));
  const tokens = phrase.match(/\{\w+\}|\[[^\]]*\]|\([^)]*\)|\S+/g) ?? [];
  const names = new Set<string>();
  const slots: Slot[] = [];
  let pattern = "^";

  for (const token of tokens) {
    if (token.startsWith("{")) {
      const name = token.slice(1, -1);
      if (!Object.hasOwn(params, name)) throw problem(`uses unknown {${name}}`);
      if (names.has(name)) throw problem(`repeats {${name}}`);
      names.add(name);
      slots.push({ name, param: params[name] });
      pattern += slotPattern(name, params[name]);
    } else if (token.startsWith("[")) {
      pattern += `(?:(?:${group(token.slice(1, -1))}) )?`;
    } else if (token.startsWith("(")) {
      pattern += `(?:${group(token.slice(1, -1))}) `;
    } else {
      pattern += escapeRegex(token) + " ";
    }
  }

  for (const name of Object.keys(params)) {
    if (!names.has(name)) throw problem(`is missing {${name}}`);
  }

  return { command, regex: new RegExp(pattern + "$"), slots };
}

function readSlot(text: string, param: Param): string | number | null {
  if (param.type === "number") return parseNumber(text);

  for (const [value, synonyms] of Object.entries(param.values)) {
    if (value === text || synonyms.includes(text)) return value;
  }
  return null;
}

function matchRule(rule: Rule, input: string): Args | null {
  const match = input.match(rule.regex);
  if (!match) return null;

  const args: Args = {};
  for (const slot of rule.slots) {
    const value = readSlot(match.groups?.[slot.name] ?? "", slot.param);
    if (value === null) return null;
    args[slot.name] = value;
  }
  return args;
}

// Numbers are returned even when out of range, for error reporting
// validate() in tools.ts decides whether they can be sent
export function createParser(tools: Tools = TOOLS): (text: unknown) => Command | null {
  const rules: Rule[] = [];
  for (const [command, spec] of Object.entries(tools)) {
    for (const phrase of spec.phrases) {
      rules.push(compile(command, phrase, spec.params));
    }
  }

  return text => {
    if (typeof text !== "string") return null;

    const input = normalize(text) + " ";
    for (const rule of rules) {
      const args = matchRule(rule, input);
      if (args) return { command: rule.command, args };
    }
    return null;
  };
}

export const parse = createParser();
