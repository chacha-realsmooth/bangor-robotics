import { defineConfig } from "vite";
import { readFileSync } from "node:fs";

const version = name => JSON.parse(readFileSync(new URL(`./node_modules/${name}/package.json`, import.meta.url))).version;

export default defineConfig({
  base: "./",
  define: {
    __VAD_VERSION__: JSON.stringify(version("@ricky0123/vad-web")),
    __ORT_VERSION__: JSON.stringify(version("onnxruntime-web")),
    __NEEDLE_VERSION__: JSON.stringify(version("cactus-needle"))
  }
});
