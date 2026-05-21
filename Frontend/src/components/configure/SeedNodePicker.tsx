import { useMemo, useState } from "react";
import { Check, ChevronsUpDown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { NodeId } from "@/lib/ppr/types";

interface SeedNodePickerProps {
  nodes: { id: NodeId }[];
  seeds: NodeId[];
  onChange: (seeds: NodeId[]) => void;
}

export function SeedNodePicker({ nodes, seeds, onChange }: SeedNodePickerProps) {
  const [open, setOpen] = useState(false);
  const seedSet = useMemo(() => new Set(seeds), [seeds]);

  const toggle = (id: NodeId) => {
    if (seedSet.has(id)) {
      onChange(seeds.filter((s) => s !== id));
    } else {
      onChange([...seeds, id]);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">Personalization seeds</Label>
        {seeds.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onChange([])}
            className="h-7 text-xs text-muted-foreground"
          >
            Clear all
          </Button>
        )}
      </div>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            <span className="truncate">
              {seeds.length === 0
                ? "Uniform (no seeds)"
                : `${seeds.length} seed${seeds.length === 1 ? "" : "s"} selected`}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search nodes…" />
            <CommandList>
              <CommandEmpty>No nodes match.</CommandEmpty>
              <CommandGroup>
                {nodes.map((n) => {
                  const selected = seedSet.has(n.id);
                  return (
                    <CommandItem
                      key={n.id}
                      value={n.id}
                      onSelect={() => toggle(n.id)}
                      aria-selected={selected}
                    >
                      <Check
                        className={cn("mr-2 h-4 w-4", selected ? "opacity-100" : "opacity-0")}
                      />
                      <span className="font-mono text-xs">{n.id}</span>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {seeds.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {seeds.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => toggle(s)}
              className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 font-mono text-xs hover:bg-muted/70"
              aria-label={`Remove seed ${s}`}
            >
              {s}
              <X className="h-3 w-3" />
            </button>
          ))}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        Empty = uniform restart over all nodes (standard PageRank).
      </p>
    </div>
  );
}
