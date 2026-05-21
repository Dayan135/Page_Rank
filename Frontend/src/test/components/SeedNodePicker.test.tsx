import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SeedNodePicker } from "@/components/configure/SeedNodePicker";

const NODES = [{ id: "A" }, { id: "B" }, { id: "C" }];

describe("<SeedNodePicker />", () => {
  it("shows the empty hint when no seeds are selected", () => {
    render(<SeedNodePicker nodes={NODES} seeds={[]} onChange={() => {}} />);
    expect(screen.getByRole("combobox")).toHaveTextContent(/uniform/i);
  });

  it("renders selected chips for current seeds", () => {
    render(<SeedNodePicker nodes={NODES} seeds={["A"]} onChange={() => {}} />);
    expect(screen.getByLabelText(/remove seed a/i)).toBeInTheDocument();
  });

  it("clears all seeds when the Clear all button is clicked", async () => {
    const onChange = vi.fn();
    render(<SeedNodePicker nodes={NODES} seeds={["A", "B"]} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: /clear all/i }));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("removes a seed when its chip is clicked", async () => {
    const onChange = vi.fn();
    render(<SeedNodePicker nodes={NODES} seeds={["A", "B"]} onChange={onChange} />);
    await userEvent.click(screen.getByLabelText(/remove seed a/i));
    expect(onChange).toHaveBeenCalledWith(["B"]);
  });
});
