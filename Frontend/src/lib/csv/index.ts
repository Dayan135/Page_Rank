import Papa from "papaparse";
import type { CSVFormat, Graph } from "@/lib/ppr/types";
import { parseEdgeList } from "./parseEdgeList";
import { parseCOO } from "./parseCOO";
import { parseAdjacency } from "./parseAdjacency";

export async function parseGraphCSV(input: File | string, format: CSVFormat): Promise<Graph> {
  const text = typeof input === "string" ? input : await input.text();

  if (format === "adjacency") {
    const result = Papa.parse<string[]>(text, {
      header: false,
      skipEmptyLines: true,
      dynamicTyping: false,
    });
    if (result.errors.length > 0) {
      throw new Error(`CSV parse error: ${result.errors[0].message}`);
    }
    return parseAdjacency(result.data);
  }

  const result = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    transformHeader: (h) => h.trim(),
  });

  if (result.errors.length > 0) {
    throw new Error(`CSV parse error: ${result.errors[0].message}`);
  }

  const headers = result.meta.fields ?? [];
  if (headers.length === 0) {
    throw new Error("CSV has no header row.");
  }

  if (format === "edge") {
    return parseEdgeList(result.data, headers);
  }
  if (format === "coo") {
    return parseCOO(result.data, headers);
  }

  throw new Error(`Unknown CSV format: ${format}`);
}
