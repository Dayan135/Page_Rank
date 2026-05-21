// Minimal toast hook (subset of shadcn/ui). Single-toast queue is enough for this app.
import * as React from "react";

type ToastVariant = "default" | "destructive";

export interface ToastInput {
  title?: string;
  description?: string;
  variant?: ToastVariant;
  durationMs?: number;
}

export interface ToastEntry extends ToastInput {
  id: string;
  open: boolean;
}

type Listener = (toasts: ToastEntry[]) => void;

let toasts: ToastEntry[] = [];
const listeners = new Set<Listener>();

function emit() {
  for (const listener of listeners) listener(toasts);
}

function genId() {
  return Math.random().toString(36).slice(2, 10);
}

export function toast(input: ToastInput) {
  const id = genId();
  const entry: ToastEntry = { id, open: true, ...input };
  toasts = [...toasts, entry];
  emit();
  const duration = input.durationMs ?? 4000;
  if (duration > 0) {
    setTimeout(() => dismiss(id), duration);
  }
  return id;
}

export function dismiss(id: string) {
  toasts = toasts.map((t) => (t.id === id ? { ...t, open: false } : t));
  emit();
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    emit();
  }, 200);
}

export function useToast() {
  const [state, setState] = React.useState<ToastEntry[]>(toasts);

  React.useEffect(() => {
    listeners.add(setState);
    return () => {
      listeners.delete(setState);
    };
  }, []);

  return { toasts: state, toast, dismiss };
}
