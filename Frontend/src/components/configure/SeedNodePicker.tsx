import { useMemo, useState } from "react";
import { Check, ChevronsUpDown, Info, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { NodeId } from "@/lib/ppr/types";

interface SeedNodePickerProps {
  nodes: { id: NodeId }[];
  seeds: NodeId[];
  onChange: (seeds: NodeId[]) => void;
  labels?: Record<NodeId, string>;
}

const MAX_SHOWN = 100;

export function SeedNodePicker({ nodes, seeds, onChange, labels }: SeedNodePickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const seedSet = useMemo(() => new Set(seeds), [seeds]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matches = q
      ? nodes.filter(
          (n) =>
            n.id.toLowerCase().includes(q) ||
            (labels?.[n.id] ?? "").toLowerCase().includes(q),
        )
      : nodes;
    return { shown: matches.slice(0, MAX_SHOWN), total: matches.length };
  }, [nodes, labels, query]);

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
        <div className="flex items-center gap-1.5">
          <Label className="text-sm">Personalization seeds</Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label="What is a seed node?"
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                <Info className="h-3.5 w-3.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs leading-relaxed">
              Seeds are the nodes the random surfer restarts from. Personalized PageRank
              measures importance <em>relative to these seeds</em> — nodes closer to them
              score higher. Pick at least one (e.g. a user, topic, or page of interest).
            </TooltipContent>
          </Tooltip>
        </div>
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
      <Popover
        open={open}
        onOpenChange={(o) => {
          setOpen(o);
          if (!o) setQuery("");
        }}
      >
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            <span className="truncate">
              {seeds.length === 0
                ? "Select at least one seed node"
                : `${seeds.length} seed${seeds.length === 1 ? "" : "s"} selected`}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput
              placeholder={labels ? "Search by name or ID…" : "Search by node ID…"}
              value={query}
              onValueChange={setQuery}
            />
            <CommandList>
              <CommandEmpty>No nodes match.</CommandEmpty>
              <CommandGroup>
                {filtered.shown.map((n) => {
                  const selected = seedSet.has(n.id);
                  const label = labels?.[n.id];
                  return (
                    <CommandItem
                      key={n.id}
                      value={label ? `${label} ${n.id}` : n.id}
                      onSelect={() => toggle(n.id)}
                      aria-selected={selected}
                    >
                      <Check
                        className={cn("mr-2 h-4 w-4", selected ? "opacity-100" : "opacity-0")}
                      />
                      {label ? (
                        <span className="flex flex-col">
                          <span className="text-sm">{label}</span>
                          <span className="font-mono text-xs text-muted-foreground">{n.id}</span>
                        </span>
                      ) : (
                        <span className="font-mono text-xs">{n.id}</span>
                      )}
                    </CommandItem>
                  );
                })}
              </CommandGroup>
              {filtered.total > MAX_SHOWN && (
                <div className="border-t bg-muted/30 p-2 text-center text-xs text-muted-foreground">
                  Showing {MAX_SHOWN} of {filtered.total.toLocaleString()} matches — type to narrow.
                </div>
              )}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {seeds.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {seeds.map((s) => {
            const label = labels?.[s];
            return (
              <button
                key={s}
                type="button"
                onClick={() => toggle(s)}
                className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-xs hover:bg-muted/70"
                aria-label={`Remove seed ${s}`}
              >
                <span className={label ? "" : "font-mono"}>{label ?? s}</span>
                <X className="h-3 w-3" />
              </button>
            );
          })}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        At least one seed is required — ranks are personalized toward the selected node(s).
      </p>
    </div>
  );
}
