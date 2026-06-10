import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { CustomMapping } from "@/lib/csv/parseCustom";

const FROM_ALIASES = ["from", "source", "src", "page_id_from", "u"];
const TO_ALIASES = ["to", "target", "dst", "page_id_to", "v"];
const LABEL_ALIASES = ["name", "label", "title", "page_title_from", "page_name"];
const WEIGHT_ALIASES = ["weight", "w", "value"];

function autoDetect(headers: string[], aliases: string[]): string {
  const lower = headers.map((h) => h.toLowerCase());
  for (const alias of aliases) {
    const idx = lower.indexOf(alias);
    if (idx !== -1) return headers[idx];
  }
  return "";
}

interface ColSelectProps {
  label: string;
  headers: string[];
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
}

function ColSelect({ label, headers, value, onChange, required }: ColSelectProps) {
  return (
    <div className="space-y-1.5">
      <Label className="text-sm">
        {label}
        {required && <span className="ml-0.5 text-destructive">*</span>}
      </Label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm",
          "ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        )}
      >
        {!required && <option value="">(none)</option>}
        {required && value === "" && <option value="" disabled>Select a column…</option>}
        {headers.map((h) => (
          <option key={h} value={h}>
            {h}
          </option>
        ))}
      </select>
    </div>
  );
}

interface ColumnMapperProps {
  headers: string[];
  previewRows: Record<string, string>[];
  onImport: (mapping: CustomMapping) => void;
}

export function ColumnMapper({ headers, previewRows, onImport }: ColumnMapperProps) {
  const [fromCol, setFromCol] = useState(() => autoDetect(headers, FROM_ALIASES));
  const [toCol, setToCol] = useState(() => autoDetect(headers, TO_ALIASES));
  const [labelCol, setLabelCol] = useState(() => autoDetect(headers, LABEL_ALIASES));
  const [weightCol, setWeightCol] = useState(() => autoDetect(headers, WEIGHT_ALIASES));

  const canImport = fromCol !== "" && toCol !== "";
  const previewCols = [fromCol, toCol, labelCol, weightCol].filter(Boolean) as string[];
  const displayCols = previewCols.length > 0 ? previewCols : headers.slice(0, 4);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <ColSelect label="Source column" headers={headers} value={fromCol} onChange={setFromCol} required />
        <ColSelect label="Target column" headers={headers} value={toCol} onChange={setToCol} required />
        <ColSelect label="Label column" headers={headers} value={labelCol} onChange={setLabelCol} />
        <ColSelect label="Weight column" headers={headers} value={weightCol} onChange={setWeightCol} />
      </div>

      {previewRows.length > 0 && displayCols.length > 0 && (
        <div className="overflow-auto rounded-md border">
          <table className="w-full text-xs">
            <thead className="bg-muted/50">
              <tr>
                {displayCols.map((col) => (
                  <th key={col} className="px-3 py-2 text-left font-mono font-medium">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {previewRows.slice(0, 3).map((row, i) => (
                <tr key={i} className="border-t">
                  {displayCols.map((col) => (
                    <td key={col} className="truncate px-3 py-1.5 font-mono max-w-[160px]">
                      {row[col] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Button
        onClick={() =>
          onImport({
            fromCol,
            toCol,
            labelCol: labelCol || undefined,
            weightCol: weightCol || undefined,
          })
        }
        disabled={!canImport}
        className="gap-2"
      >
        Import
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  );
}
