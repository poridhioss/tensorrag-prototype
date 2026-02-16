import { useEffect, useRef } from "react";
import { saveProjectCard } from "@/lib/api";
import { useEditorStore } from "@/store/editorStore";
import { useWorkspaceStore } from "@/store/workspaceStore";

/**
 * Debounced auto-sync: saves the active card file to S3 (project-scoped)
 * 1 second after edits stop.
 */
export function useCardAutoSync() {
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const markFileSaved = useEditorStore((s) => s.markFileSaved);
  const setIsSyncing = useEditorStore((s) => s.setIsSyncing);

  const activeProject = useWorkspaceStore((s) => s.activeProject);

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Find the active file's content fingerprint
  const activeFile = cardFiles.find((f) => f.path === activeFilePath);
  const content = activeFile?.content;
  const isDirty = activeFile?.isDirty;
  const filePath = activeFile?.path;

  useEffect(() => {
    if (!isDirty || !activeFilePath || !content || !filePath || !activeProject) return;

    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(async () => {
      try {
        setIsSyncing(true);
        await saveProjectCard(activeProject, filePath, content);
        markFileSaved(activeFilePath);
      } catch {
        // Sync failure is non-critical — file stays dirty for retry
      } finally {
        setIsSyncing(false);
      }
    }, 1000);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [content, isDirty, activeFilePath, filePath, activeProject, markFileSaved, setIsSyncing]);
}
