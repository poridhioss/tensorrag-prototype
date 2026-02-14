"use client";

import { ThemeToggle } from "@/components/ThemeToggle";

export function Header() {
  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-border bg-bg-secondary shrink-0">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-sm tracking-tight">TensorRag</span>
      </div>
      <div className="flex items-center gap-3">
        <ThemeToggle />
      </div>
    </header>
  );
}
