import { describe, it, expect, vi } from "vitest";
import type { ReactElement } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SeedNodePicker } from "@/components/configure/SeedNodePicker";

const NODES = [{ id: "A" }, { id: "B" }, { id: "C" }];

// The seed label carries an info Tooltip, which needs a TooltipProvider ancestor
// (App.tsx provides one app-wide).
function renderPicker(ui: ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>);
}

describe("<SeedNodePicker />", () => {
  it("shows the empty-state hint when no seeds are selected", () => {
    renderPicker(<SeedNodePicker nodes={NODES} seeds={[]} onChange={() => {}} />);
    expect(screen.getByRole("combobox")).toHaveTextContent(/select at least one seed/i);
  });

  it("renders selected chips for current seeds", () => {
    renderPicker(<SeedNodePicker nodes={NODES} seeds={["A"]} onChange={() => {}} />);
    expect(screen.getByLabelText(/remove seed a/i)).toBeInTheDocument();
  });

  it("clears all seeds when the Clear all button is clicked", async () => {
    const onChange = vi.fn();
    renderPicker(<SeedNodePicker nodes={NODES} seeds={["A", "B"]} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("removes a seed when its chip is clicked", async () => {
    const onChange = vi.fn();
    renderPicker(<SeedNodePicker nodes={NODES} seeds={["A", "B"]} onChange={onChange} />);
    await userEvent.click(screen.getByLabelText(/remove seed a/i));
    expect(onChange).toHaveBeenCalledWith(["B"]);
  });

  it("filters the node list by ID as the user types", async () => {
    renderPicker(<SeedNodePicker nodes={NODES} seeds={[]} onChange={() => {}} />);
    await userEvent.click(screen.getByRole("combobox"));
    await userEvent.type(screen.getByPlaceholderText(/search by node id/i), "B");
    expect(screen.getByRole("option", { name: /b/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /^a$/i })).not.toBeInTheDocument();
  });

  it("filters by label (name) when labels are provided", async () => {
    const labels = { "10": "Anarchism", "20": "Atheism", "30": "Biology" };
    const nodes = [{ id: "10" }, { id: "20" }, { id: "30" }];
    renderPicker(
      <SeedNodePicker nodes={nodes} seeds={[]} onChange={() => {}} labels={labels} />,
    );
    await userEvent.click(screen.getByRole("combobox"));
    await userEvent.type(screen.getByPlaceholderText(/search by name or id/i), "anarch");
    expect(screen.getByRole("option", { name: /anarchism/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /biology/i })).not.toBeInTheDocument();
  });

  it("caps the rendered list and shows a truncation hint on large graphs", async () => {
    const bigNodes = Array.from({ length: 250 }, (_, i) => ({ id: `node-${i}` }));
    renderPicker(<SeedNodePicker nodes={bigNodes} seeds={[]} onChange={() => {}} />);
    await userEvent.click(screen.getByRole("combobox"));
    expect(screen.getAllByRole("option")).toHaveLength(100);
    expect(screen.getByText(/showing 100 of 250 matches/i)).toBeInTheDocument();
  });
});
