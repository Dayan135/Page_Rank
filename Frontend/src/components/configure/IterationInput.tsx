import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface IterationInputProps {
  maxIter: number;
  tolerance: number;
  onMaxIterChange: (n: number) => void;
  onToleranceChange: (n: number) => void;
}

export function IterationInput({
  maxIter,
  tolerance,
  onMaxIterChange,
  onToleranceChange,
}: IterationInputProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="space-y-2">
        <Label htmlFor="max-iter" className="text-sm">
          Max iterations
        </Label>
        <Input
          id="max-iter"
          type="number"
          inputMode="numeric"
          min={1}
          max={10000}
          step={1}
          value={maxIter}
          onChange={(e) => {
            const n = Number(e.target.value);
            if (Number.isInteger(n) && n >= 1 && n <= 10000) onMaxIterChange(n);
          }}
          className="font-mono tabular-nums"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="tolerance" className="text-sm">
          Tolerance
        </Label>
        <Input
          id="tolerance"
          type="number"
          inputMode="decimal"
          min={1e-12}
          step={1e-6}
          value={tolerance}
          onChange={(e) => {
            const n = Number(e.target.value);
            if (Number.isFinite(n) && n > 0) onToleranceChange(n);
          }}
          className="font-mono tabular-nums"
        />
      </div>
    </div>
  );
}
