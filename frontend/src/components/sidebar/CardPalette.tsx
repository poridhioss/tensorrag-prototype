"use client";

import { useState } from "react";
import { usePipelineStore } from "@/store/pipelineStore";
import { useWorkspaceStore } from "@/store/workspaceStore";
import type { CardSchema } from "@/lib/types";
import { ProjectSelector } from "@/components/editor/ProjectSelector";
import { ChevronRight, FolderOpen } from "lucide-react";

/** Build a folder → cards mapping from cardPathMap + cardSchemas. */
function groupByFolder(
  cardSchemas: CardSchema[],
  cardPathMap: Record<string, string>
): { folder: string; cards: CardSchema[] }[] {
  const folderMap = new Map<string, CardSchema[]>();

  for (const schema of cardSchemas) {
    const filePath = cardPathMap[schema.card_type];
    let folder: string;

    if (filePath) {
      // e.g. "data/data_load.py" → "data", "model.py" → "" (root)
      const lastSlash = filePath.lastIndexOf("/");
      folder = lastSlash >= 0 ? filePath.substring(0, lastSlash) : "";
    } else {
      folder = "";
    }

    if (!folderMap.has(folder)) {
      folderMap.set(folder, []);
    }
    folderMap.get(folder)!.push(schema);
  }

  // Sort folders alphabetically, root ("") first
  const folders = Array.from(folderMap.keys()).sort((a, b) => {
    if (a === "") return -1;
    if (b === "") return 1;
    return a.localeCompare(b);
  });

  return folders.map((folder) => ({
    folder,
    cards: folderMap.get(folder)!,
  }));
}

export function CardPalette() {
  const cardSchemas = usePipelineStore((s) => s.cardSchemas);
  const activeProject = useWorkspaceStore((s) => s.activeProject);
  const cardPathMap = useWorkspaceStore((s) => s.cardPathMap);

  const [collapsedFolders, setCollapsedFolders] = useState<Set<string>>(
    new Set()
  );

  const groups = groupByFolder(cardSchemas, cardPathMap);

  const toggleFolder = (folder: string) => {
    setCollapsedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folder)) {
        next.delete(folder);
      } else {
        next.add(folder);
      }
      return next;
    });
  };

  const onDragStart = (e: React.DragEvent, cardType: string) => {
    e.dataTransfer.setData("application/tensorrag-card", cardType);
    e.dataTransfer.effectAllowed = "move";
  };

  return (
    <aside className="h-full border-r border-border bg-bg-secondary overflow-y-auto flex flex-col">
      {/* Project selector */}
      <ProjectSelector allowCreate={false} />

      {/* Card list */}
      <div className="flex-1 overflow-y-auto p-3">
        <h2 className="text-xs font-semibold uppercase text-text-secondary mb-3">
          Cards
        </h2>

        {!activeProject && (
          <div className="text-xs text-text-secondary text-center py-4">
            Select or create a project to see cards.
          </div>
        )}

        {activeProject &&
          groups.map(({ folder, cards }) => {
            const isRoot = folder === "";
            const isCollapsed = collapsedFolders.has(folder);
            const folderLabel = isRoot
              ? "Root"
              : folder.split("/").pop() || folder;

            return (
              <div key={folder || "__root__"} className="mb-3">
                {/* Folder header */}
                <button
                  onClick={() => toggleFolder(folder)}
                  className="flex items-center gap-1 w-full text-left mb-1.5 group/folder"
                >
                  <ChevronRight
                    size={12}
                    className={`text-text-secondary transition-transform duration-150 ${
                      isCollapsed ? "" : "rotate-90"
                    }`}
                  />
                  <FolderOpen size={12} className="text-accent shrink-0" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary group-hover/folder:text-foreground transition-colors">
                    {folderLabel}
                  </span>
                  <span className="text-[9px] text-text-secondary/60 ml-auto">
                    {cards.length}
                  </span>
                </button>

                {/* Cards in folder */}
                {!isCollapsed && (
                  <div className="space-y-1.5 ml-1 pl-3 border-l border-border/40">
                    {cards.map((schema) => (
                      <div
                        key={schema.card_type}
                        draggable
                        onDragStart={(e) => onDragStart(e, schema.card_type)}
                        className="group rounded-lg px-2.5 py-2
                          bg-background cursor-grab active:cursor-grabbing
                          border border-border hover:border-text-secondary/40
                          hover:bg-bg-secondary/60 hover:shadow-md
                          active:scale-[0.97] transition-all duration-150"
                      >
                        <div className="text-xs font-medium group-hover:text-text-primary transition-colors">
                          {schema.display_name}
                        </div>
                        <div className="text-[10px] text-text-secondary truncate mt-0.5">
                          {schema.description}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}

        {activeProject && cardSchemas.length === 0 && (
          <div className="text-xs text-text-secondary text-center py-4">
            No cards in this project yet.
            <br />
            Switch to Editor to create cards.
          </div>
        )}
      </div>
    </aside>
  );
}
