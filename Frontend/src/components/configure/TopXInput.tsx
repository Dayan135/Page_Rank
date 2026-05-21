import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface TopXInputProps {
  value: number;
  max: number;
  onChange: (n: number) => void;
}

export function TopXInput({ value, max, onChange }: TopXInputProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="top-x" className="text-sm">
        Top X pages
      </Label>
      <Input
        id="top-x"
        type="number"
        inputMode="numeric"
        min={1}
        max={max}
        step={1}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isInteger(n) && n >= 1 && n <= max) onChange(n);
        }}
        className="font-mono tabular-nums"
      />
      <p className="text-xs text-muted-foreground">Highlight the top-ranked nodes in summary cards.</p>
    </div>
  );
}
