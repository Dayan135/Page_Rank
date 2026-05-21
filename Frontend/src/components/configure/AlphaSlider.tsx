import { useState, useEffect } from "react";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface AlphaSliderProps {
  value: number;
  onChange: (value: number) => void;
}

export function AlphaSlider({ value, onChange }: AlphaSliderProps) {
  const [text, setText] = useState(value.toFixed(2));
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    setText(value.toFixed(2));
    setInvalid(false);
  }, [value]);

  const commit = (raw: string) => {
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0 || n > 1) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    onChange(Number(n.toFixed(4)));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor="alpha-input" className="text-sm">
          Damping factor (α)
        </Label>
        <Input
          id="alpha-input"
          type="number"
          inputMode="decimal"
          step={0.01}
          min={0}
          max={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={(e) => commit(e.target.value)}
          aria-invalid={invalid}
          className={cn(
            "h-9 w-24 text-right font-mono tabular-nums",
            invalid && "border-destructive ring-1 ring-destructive",
          )}
        />
      </div>
      <Slider
        value={[value]}
        min={0}
        max={1}
        step={0.01}
        onValueChange={(arr) => onChange(arr[0])}
        aria-label="Damping factor"
      />
      <p className="text-xs text-muted-foreground">
        Higher α = more weight on graph structure; (1 − α) = teleport probability.
      </p>
    </div>
  );
}
