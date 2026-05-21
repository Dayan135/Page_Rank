import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormatPicker } from "@/components/upload/FormatPicker";

describe("<FormatPicker />", () => {
  it("renders three radio options", () => {
    render(<FormatPicker value={null} onChange={() => {}} />);
    expect(screen.getByText("Edge list")).toBeInTheDocument();
    expect(screen.getByText("COO triplets")).toBeInTheDocument();
    expect(screen.getByText("Adjacency matrix")).toBeInTheDocument();
  });

  it("calls onChange when the user selects a format", async () => {
    const onChange = vi.fn();
    render(<FormatPicker value={null} onChange={onChange} />);
    await userEvent.click(screen.getByText("Edge list"));
    expect(onChange).toHaveBeenCalledWith("edge");
  });

  it("marks the selected option with aria-checked=true", () => {
    render(<FormatPicker value="coo" onChange={() => {}} />);
    const radios = screen.getAllByRole("radio");
    const coo = radios.find((r) => r.getAttribute("value") === "coo");
    expect(coo).toHaveAttribute("aria-checked", "true");
  });
});
