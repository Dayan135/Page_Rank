import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AlphaSlider } from "@/components/configure/AlphaSlider";

describe("<AlphaSlider />", () => {
  it("calls onChange when the numeric input commits a valid value", async () => {
    const onChange = vi.fn();
    render(<AlphaSlider value={0.85} onChange={onChange} />);
    const input = screen.getByRole("spinbutton", { name: /damping factor/i });
    await userEvent.clear(input);
    await userEvent.type(input, "0.5");
    fireEvent.blur(input);
    expect(onChange).toHaveBeenCalledWith(0.5);
  });

  it("flags an invalid value (> 1) instead of calling onChange", async () => {
    const onChange = vi.fn();
    render(<AlphaSlider value={0.85} onChange={onChange} />);
    const input = screen.getByRole("spinbutton", { name: /damping factor/i });
    await userEvent.clear(input);
    await userEvent.type(input, "1.5");
    fireEvent.blur(input);
    expect(onChange).not.toHaveBeenCalled();
    expect(input).toHaveAttribute("aria-invalid", "true");
  });

  it("renders the value with two decimal places", () => {
    render(<AlphaSlider value={0.85} onChange={() => {}} />);
    const input = screen.getByRole("spinbutton", {
      name: /damping factor/i,
    }) as HTMLInputElement;
    expect(input.value).toBe("0.85");
  });
});
