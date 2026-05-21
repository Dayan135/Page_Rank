import { NavLink } from "react-router-dom";
import { Network, Settings2, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";
import { useAppStore } from "@/store/useAppStore";

const NAV_ITEMS = [
  { to: "/upload", label: "Upload", icon: Network },
  { to: "/configure", label: "Configure", icon: Settings2 },
  { to: "/results", label: "Results", icon: BarChart3 },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const graph = useAppStore((s) => s.graph);
  const result = useAppStore((s) => s.result);

  return (
    <div className="min-h-dvh flex flex-col bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur">
        <div className="container flex h-14 items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="grid place-items-center h-8 w-8 rounded-md bg-primary text-primary-foreground font-mono font-semibold">
              P
            </div>
            <div className="font-mono text-base font-semibold tracking-tight">PPR Analyzer</div>
          </div>
          <nav className="flex items-center gap-1" aria-label="Primary">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const disabled =
                (to === "/configure" && !graph) || (to === "/results" && !result);
              return (
                <NavLink
                  key={to}
                  to={to}
                  aria-disabled={disabled}
                  onClick={(e) => {
                    if (disabled) e.preventDefault();
                  }}
                  className={({ isActive }) =>
                    cn(
                      "inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      disabled && "pointer-events-none opacity-50",
                    )
                  }
                >
                  <Icon className="h-4 w-4" aria-hidden />
                  {label}
                </NavLink>
              );
            })}
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main className="flex-1">
        <div className="container py-8">{children}</div>
      </main>
      <footer className="border-t">
        <div className="container py-4 text-xs text-muted-foreground font-mono">
          PPR Analyzer · client-side mock · swap-in real CUDA backend via{" "}
          <span className="text-foreground">src/lib/ppr/adapter.ts</span>
        </div>
      </footer>
    </div>
  );
}
