import { Button } from "@/components/ui/button";
import type { CSVFormat } from "@/lib/ppr/types";

interface SampleGraphPickerProps {
  onLoad: (file: { name: string; text: string; format: CSVFormat }) => void;
  disabled?: boolean;
}

const SAMPLES: { name: string; path: string; format: CSVFormat; label: string }[] = [
  { name: "karate-edgelist.csv", path: "/samples/karate-edgelist.csv", format: "edge", label: "Zachary's karate club (edge list)" },
  { name: "small-coo.csv", path: "/samples/small-coo.csv", format: "coo", label: "Small COO matrix" },
  { name: "tiny-adjacency.csv", path: "/samples/tiny-adjacency.csv", format: "adjacency", label: "Tiny adjacency matrix" },
  { name: "enwiki-2002.csv", path: "/samples/enwiki-2002.csv", format: "custom-edge-list", label: "English Wikipedia 2002 (224k links)" },
];

export function SampleGraphPicker({ onLoad, disabled }: SampleGraphPickerProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-muted-foreground">Or try a sample:</span>
      {SAMPLES.map((s) => (
        <Button
          key={s.path}
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={async () => {
            const res = await fetch(s.path);
            if (!res.ok) return;
            const text = await res.text();
            onLoad({ name: s.name, text, format: s.format });
          }}
        >
          {s.label}
        </Button>
      ))}
    </div>
  );
}
