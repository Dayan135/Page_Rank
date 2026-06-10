import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { CSVFormat } from "@/lib/ppr/types";

interface FormatOption {
  value: CSVFormat;
  title: string;
  description: string;
  example: string;
}

const FORMATS: FormatOption[] = [
  {
    value: "edge",
    title: "Edge list",
    description: "Three columns: source, target, weight (optional).",
    example: "source,target,weight\nA,B,1.0\nA,C,0.5\nB,C,1.0",
  },
  {
    value: "coo",
    title: "COO triplets",
    description: "Integer indices: row_idx, col_idx, value.",
    example: "row_idx,col_idx,value\n0,1,1.0\n0,2,0.5\n1,2,1.0",
  },
  {
    value: "adjacency",
    title: "Adjacency matrix",
    description: "N×N grid, first row & col are node labels.",
    example: ",A,B,C\nA,0,1,0.5\nB,0,0,1\nC,1,0,0",
  },
  {
    value: "custom-edge-list",
    title: "Custom edge list",
    description: "Any CSV/TSV edge list — you map columns after uploading.",
    example: "page_id_from,page_id_to,page_title_from\n1,2,Cat\n2,3,Animal\n3,1,Life",
  },
];

interface FormatPickerProps {
  value: CSVFormat | null;
  onChange: (format: CSVFormat) => void;
}

export function FormatPicker({ value, onChange }: FormatPickerProps) {
  return (
    <RadioGroup
      value={value ?? ""}
      onValueChange={(v) => onChange(v as CSVFormat)}
      className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"
      aria-label="Choose a CSV format"
    >
      {FORMATS.map((opt) => {
        const id = `format-${opt.value}`;
        const selected = value === opt.value;
        return (
          <Label
            key={opt.value}
            htmlFor={id}
            className={cn(
              "flex cursor-pointer flex-col gap-2 rounded-lg border bg-card p-4 transition-colors",
              "hover:bg-muted/40",
              selected ? "border-primary ring-2 ring-primary/30" : "border-border",
            )}
          >
            <div className="flex items-center gap-2">
              <RadioGroupItem id={id} value={opt.value} />
              <span className="text-sm font-semibold">{opt.title}</span>
            </div>
            <p className="text-xs text-muted-foreground">{opt.description}</p>
            <pre className="overflow-auto rounded bg-muted p-2 font-mono text-[11px] leading-tight">
              {opt.example}
            </pre>
          </Label>
        );
      })}
    </RadioGroup>
  );
}
