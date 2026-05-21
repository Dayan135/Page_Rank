import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatScore(value: number, digits = 6): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 1e-4) return value.toExponential(2);
  return value.toFixed(digits);
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat(undefined).format(value);
}
