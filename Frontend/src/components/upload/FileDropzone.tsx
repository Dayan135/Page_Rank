import { useCallback, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileDropzoneProps {
  disabled: boolean;
  onFile: (file: File) => void;
  onError: (message: string) => void;
}

function isAcceptedFile(file: File): boolean {
  if (file.type === "text/csv" || file.type === "application/vnd.ms-excel") return true;
  return file.name.toLowerCase().endsWith(".csv");
}

export function FileDropzone({ disabled, onFile, onError }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isOver, setIsOver] = useState(false);

  const accept = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      if (files.length > 1) {
        onError("Multiple files were dropped — only the first will be used.");
      }
      const file = files[0];
      if (!isAcceptedFile(file)) {
        onError(`"${file.name}" is not a CSV file.`);
        return;
      }
      onFile(file);
    },
    [onFile, onError],
  );

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsOver(false);
    if (disabled) return;
    accept(e.dataTransfer.files);
  };

  const handleKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload CSV file"
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={handleKey}
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-10 text-center transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        disabled && "cursor-not-allowed opacity-50",
        !disabled && "cursor-pointer hover:bg-muted/40",
        isOver && !disabled ? "border-primary bg-muted/60" : "border-border",
      )}
    >
      <UploadCloud className="h-10 w-10 text-muted-foreground" aria-hidden />
      <div className="space-y-1">
        <p className="text-base font-medium">
          {disabled ? "Pick a CSV format above first" : "Drop your CSV here, or click to browse"}
        </p>
        <p className="text-xs text-muted-foreground">.csv files only · single file</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        disabled={disabled}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
        data-testid="file-input"
      />
    </div>
  );
}
