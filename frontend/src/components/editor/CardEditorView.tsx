"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import Editor from "@monaco-editor/react";
import { useTheme } from "next-themes";
import { useEditorStore } from "@/store/editorStore";
import { useWorkspaceStore } from "@/store/workspaceStore";
import { listProjectCards, getCardSource } from "@/lib/api";
import { setupBaseCardIntelliSense } from "@/lib/services/monacoCardIntelliSense";
import { useCardAutoSync } from "@/hooks/useCardAutoSync";
import { CardFileTree } from "./CardFileTree";
import { EditorToolbar } from "./EditorToolbar";
import { ValidationPanel } from "./ValidationPanel";
import type { CardFile } from "@/store/editorStore";

const LEFT_MIN = 160;
const LEFT_MAX = 400;
const LEFT_DEFAULT = 208;
const RIGHT_MIN = 200;
const RIGHT_MAX = 480;
const RIGHT_DEFAULT = 288;

type ResizeTarget = "left" | "right" | null;

export function CardEditorView() {
  const { theme } = useTheme();
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const updateFileContent = useEditorStore((s) => s.updateFileContent);
  const setCardFiles = useEditorStore((s) => s.setCardFiles);
  const setActiveFilePath = useEditorStore((s) => s.setActiveFilePath);
  const setIsLoadingFiles = useEditorStore((s) => s.setIsLoadingFiles);
  const isLoadingFiles = useEditorStore((s) => s.isLoadingFiles);

  const activeProject = useWorkspaceStore((s) => s.activeProject);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const editorRef = useRef<any>(null);

  // Resizable panel widths
  const [leftWidth, setLeftWidth] = useState(LEFT_DEFAULT);
  const [rightWidth, setRightWidth] = useState(RIGHT_DEFAULT);
  const [resizeTarget, setResizeTarget] = useState<ResizeTarget>(null);
  const resizeStartX = useRef(0);
  const resizeStartWidth = useRef(0);

  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  const activeFile = cardFiles.find((f) => f.path === activeFilePath);

  // Auto-sync hook
  useCardAutoSync();

  // Resize handlers
  useEffect(() => {
    if (!resizeTarget) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - resizeStartX.current;

      if (resizeTarget === "left") {
        const newWidth = Math.min(LEFT_MAX, Math.max(LEFT_MIN, resizeStartWidth.current + delta));
        setLeftWidth(newWidth);
      } else {
        // For the right panel, dragging left makes it bigger
        const newWidth = Math.min(RIGHT_MAX, Math.max(RIGHT_MIN, resizeStartWidth.current - delta));
        setRightWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setResizeTarget(null);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [resizeTarget]);

  function startResize(target: "left" | "right", e: React.MouseEvent) {
    e.preventDefault();
    resizeStartX.current = e.clientX;
    resizeStartWidth.current = target === "left" ? leftWidth : rightWidth;
    setResizeTarget(target);
  }

  // Load card files from active project
  useEffect(() => {
    if (!activeProject) {
      setCardFiles([]);
      return;
    }

    let cancelled = false;

    async function loadFiles() {
      setIsLoadingFiles(true);
      try {
        const entries = await listProjectCards(activeProject!);
        if (cancelled) return;

        const files: CardFile[] = entries
          .filter((e) => e.type === "file" && e.path.endsWith(".py"))
          .map((e) => {
            const name = e.path.split("/").pop() || e.path;
            return {
              name,
              path: e.path,
              content: "",
              language: "python" as const,
              isDirty: false,
            };
          });

        const folderKeeps: CardFile[] = entries
          .filter((e) => e.type === "folder")
          .map((e) => ({
            name: ".keep",
            path: `${e.path}/.keep`,
            content: "",
            language: "python" as const,
            isDirty: false,
          }));

        setCardFiles([...files, ...folderKeeps]);
        setActiveFilePath(null);
      } catch {
        if (!cancelled) setCardFiles([]);
      } finally {
        if (!cancelled) setIsLoadingFiles(false);
      }
    }

    loadFiles();
    return () => {
      cancelled = true;
    };
  }, [activeProject]); // eslint-disable-line react-hooks/exhaustive-deps

  // Lazy-load card source when user selects a file with empty content
  useEffect(() => {
    if (!activeFilePath || !activeProject) return;
    const file = cardFiles.find((f) => f.path === activeFilePath);
    if (!file || file.content !== "" || file.isDirty) return;

    let cancelled = false;

    async function loadSource() {
      try {
        const result = await getCardSource(activeProject!, activeFilePath!);
        if (cancelled) return;
        const current = useEditorStore.getState().cardFiles;
        useEditorStore.getState().setCardFiles(
          current.map((f) =>
            f.path === activeFilePath ? { ...f, content: result.source_code } : f
          )
        );
      } catch {
        // File source fetch failed — leave empty
      }
    }

    loadSource();
    return () => { cancelled = true; };
  }, [activeFilePath, activeProject]); // eslint-disable-line react-hooks/exhaustive-deps

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleEditorDidMount = useCallback((editor: any, monaco: any) => {
    editorRef.current = editor;

    monaco.editor.defineTheme("tensorrag-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "", foreground: "d4d4d4" },
        { token: "comment", foreground: "6A9955" },
        { token: "keyword", foreground: "569CD6" },
        { token: "string", foreground: "CE9178" },
        { token: "number", foreground: "B5CEA8" },
        { token: "type", foreground: "4EC9B0" },
        { token: "function", foreground: "DCDCAA" },
      ],
      colors: {
        "editor.background": "#0d1117",
        "editor.foreground": "#d4d4d4",
        "editor.lineHighlightBackground": "#161b22",
        "editor.selectionBackground": "#264f78",
        "editorCursor.foreground": "#ffffff",
        "editorLineNumber.foreground": "#6e7681",
      },
    });

    monaco.editor.defineTheme("tensorrag-light", {
      base: "vs",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#f8fafc",
        "editor.foreground": "#1e293b",
        "editor.lineHighlightBackground": "#f1f5f9",
        "editorLineNumber.foreground": "#94a3b8",
      },
    });

    setupBaseCardIntelliSense(monaco);
    monaco.editor.setTheme(isDark ? "tensorrag-dark" : "tensorrag-light");
  }, [isDark]);

  useEffect(() => {
    if (editorRef.current) {
      const monaco = (window as any).monaco; // eslint-disable-line @typescript-eslint/no-explicit-any
      if (monaco) {
        monaco.editor.setTheme(isDark ? "tensorrag-dark" : "tensorrag-light");
      }
    }
  }, [isDark]);

  function handleEditorChange(value: string | undefined) {
    if (value !== undefined && activeFilePath) {
      updateFileContent(activeFilePath, value);
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <EditorToolbar />

      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: file tree */}
        <div
          className="shrink-0 border-r border-border overflow-hidden"
          style={{ width: `${leftWidth}px` }}
        >
          <CardFileTree />
        </div>

        {/* Left resize handle */}
        <div
          onMouseDown={(e) => startResize("left", e)}
          className={`w-1 shrink-0 cursor-col-resize transition-colors relative group
            ${resizeTarget === "left" ? "bg-accent/50" : "hover:bg-accent/30"}`}
        >
          <div className="absolute inset-y-0 -left-1 -right-1" />
        </div>

        {/* Center panel: editor */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          {activeFile ? (
            <Editor
              height="100%"
              language="python"
              theme={isDark ? "tensorrag-dark" : "tensorrag-light"}
              value={activeFile.content}
              onChange={handleEditorChange}
              onMount={handleEditorDidMount}
              options={{
                fontSize: 14,
                fontFamily:
                  '"JetBrains Mono", "Fira Code", "SF Mono", Monaco, Consolas, monospace',
                fontLigatures: true,
                minimap: { enabled: false },
                lineNumbers: "on",
                wordWrap: "on",
                tabSize: 4,
                insertSpaces: true,
                scrollBeyondLastLine: false,
                automaticLayout: true,
                cursorBlinking: "smooth",
                bracketPairColorization: { enabled: true },
                suggestOnTriggerCharacters: true,
                quickSuggestions: {
                  other: true,
                  comments: false,
                  strings: false,
                },
                parameterHints: { enabled: true },
                hover: { enabled: true, delay: 300 },
                snippetSuggestions: "top",
              }}
              loading={
                <div className="flex-1 flex items-center justify-center">
                  <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                </div>
              }
            />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-text-secondary">
                <svg
                  className="w-12 h-12 mx-auto mb-3 text-text-secondary/40"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5"
                  />
                </svg>
                <p className="text-sm">
                  {isLoadingFiles
                    ? "Loading card files..."
                    : !activeProject
                      ? "Select a project to start editing"
                      : "Create or select a card file to start editing"}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Right resize handle */}
        <div
          onMouseDown={(e) => startResize("right", e)}
          className={`w-1 shrink-0 cursor-col-resize transition-colors relative group
            ${resizeTarget === "right" ? "bg-accent/50" : "hover:bg-accent/30"}`}
        >
          <div className="absolute inset-y-0 -left-1 -right-1" />
        </div>

        {/* Right panel: validation */}
        <div
          className="shrink-0 border-l border-border overflow-hidden"
          style={{ width: `${rightWidth}px` }}
        >
          <ValidationPanel />
        </div>
      </div>
    </div>
  );
}
