/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Backend target for the dev proxy. Point at a local uvicorn or an SSH tunnel
// to a remote GPU host (see Backend/server/README.md).
const PPR_TARGET = process.env.VITE_PPR_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      // Frontend calls /api/* → forwarded to the FastAPI PPR backend.
      "/api": { target: PPR_TARGET, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "happy-dom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    coverage: {
      provider: "v8",
      include: ["src/lib/**", "src/store/**"],
      reporter: ["text", "html"],
    },
  },
});
