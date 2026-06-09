/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute base URL for the PPR backend. Defaults to "/api" (Vite-proxied in dev). */
  readonly VITE_PPR_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
