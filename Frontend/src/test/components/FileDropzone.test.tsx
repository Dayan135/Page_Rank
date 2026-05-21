import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FileDropzone } from "@/components/upload/FileDropzone";

describe("<FileDropzone />", () => {
  it("is non-interactive when disabled", () => {
    render(<FileDropzone disabled={true} onFile={() => {}} onError={() => {}} />);
    const zone = screen.getByRole("button", { name: /upload csv file/i });
    expect(zone).toHaveAttribute("aria-disabled", "true");
    expect(zone).toHaveAttribute("tabindex", "-1");
  });

  it("calls onFile when the user picks a CSV via the file input", async () => {
    const onFile = vi.fn();
    render(<FileDropzone disabled={false} onFile={onFile} onError={() => {}} />);
    const file = new File(["source,target\nA,B\n"], "graph.csv", { type: "text/csv" });
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    await userEvent.upload(input, file, { applyAccept: false });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("rejects non-CSV files with an error message", async () => {
    const onFile = vi.fn();
    const onError = vi.fn();
    render(<FileDropzone disabled={false} onFile={onFile} onError={onError} />);
    const file = new File(["nope"], "image.png", { type: "image/png" });
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    await userEvent.upload(input, file, { applyAccept: false });
    expect(onFile).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(expect.stringMatching(/not a csv/i));
  });
});
