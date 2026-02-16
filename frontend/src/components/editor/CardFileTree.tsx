"use client";

import { useState } from "react";
import { useEditorStore, type CardFile } from "@/store/editorStore";
import { useWorkspaceStore } from "@/store/workspaceStore";
import { generateCardTemplate } from "@/lib/cardTemplate";
import { deleteProjectCard, deleteProjectFolder, createProjectFolder } from "@/lib/api";
import {
  Plus,
  FileCode2,
  Trash2,
  FolderPlus,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { ProjectSelector } from "./ProjectSelector";

interface TreeNode {
  name: string;
  path: string;
  type: "file" | "folder";
  children: TreeNode[];
}

function buildTree(files: CardFile[]): TreeNode[] {
  const root: TreeNode[] = [];

  for (const file of files) {
    const parts = file.path.split("/");
    let current = root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLast = i === parts.length - 1;
      const pathSoFar = parts.slice(0, i + 1).join("/");

      let existing = current.find((n) => n.name === part);

      if (!existing) {
        existing = {
          name: part,
          path: pathSoFar,
          type: isLast && file.path.endsWith(".py") ? "file" : "folder",
          children: [],
        };
        current.push(existing);
      }

      if (!isLast) {
        current = existing.children;
      }
    }
  }

  // Sort: folders first, then files, alphabetically within each
  function sortNodes(nodes: TreeNode[]) {
    nodes.sort((a, b) => {
      if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    for (const n of nodes) {
      if (n.children.length > 0) sortNodes(n.children);
    }
  }
  sortNodes(root);
  return root;
}

export function CardFileTree() {
  const cardFiles = useEditorStore((s) => s.cardFiles);
  const activeFilePath = useEditorStore((s) => s.activeFilePath);
  const setActiveFilePath = useEditorStore((s) => s.setActiveFilePath);
  const addCardFile = useEditorStore((s) => s.addCardFile);
  const removeCardFile = useEditorStore((s) => s.removeCardFile);
  const isLoadingFiles = useEditorStore((s) => s.isLoadingFiles);

  const activeProject = useWorkspaceStore((s) => s.activeProject);

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set()
  );
  const [showNewCardDialog, setShowNewCardDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [newItemParent, setNewItemParent] = useState<string>("");
  const [newCardName, setNewCardName] = useState("");
  const [newFolderName, setNewFolderName] = useState("");
  const [hoveredFile, setHoveredFile] = useState<string | null>(null);

  const tree = buildTree(cardFiles);

  function toggleFolder(path: string) {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function handleCreateCard() {
    const name = newCardName.trim();
    if (!name) return;

    const fileName = name.endsWith(".py") ? name : `${name}.py`;
    const cardType = fileName.replace(/\.py$/, "");
    const displayName = cardType
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");

    const fullPath = newItemParent
      ? `${newItemParent}/${fileName}`
      : fileName;

    if (cardFiles.some((f) => f.path === fullPath)) {
      alert(`A card file at "${fullPath}" already exists.`);
      return;
    }

    const content = generateCardTemplate(cardType, displayName);
    const file: CardFile = {
      name: fileName,
      path: fullPath,
      content,
      language: "python",
      isDirty: false,
    };

    addCardFile(file);
    setNewCardName("");
    setShowNewCardDialog(false);

    // Auto-expand parent folder
    if (newItemParent) {
      setExpandedFolders((prev) => new Set([...prev, newItemParent]));
    }
  }

  async function handleCreateFolder() {
    const name = newFolderName.trim();
    if (!name || !activeProject) return;

    const fullPath = newItemParent ? `${newItemParent}/${name}` : name;

    try {
      await createProjectFolder(activeProject, fullPath);
      // Add a placeholder so the folder shows up in the tree
      const keepFile: CardFile = {
        name: ".keep",
        path: `${fullPath}/.keep`,
        content: "",
        language: "python",
        isDirty: false,
      };
      addCardFile(keepFile);
      setExpandedFolders((prev) => new Set([...prev, fullPath]));
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to create folder");
    }

    setNewFolderName("");
    setShowNewFolderDialog(false);
  }

  async function handleDeleteFile(path: string) {
    const file = cardFiles.find((f) => f.path === path);
    if (!file || !activeProject) return;

    if (!confirm(`Delete "${file.path}"?`)) return;

    try {
      await deleteProjectCard(activeProject, path);
    } catch {
      // File might not be published yet — that's fine
    }

    removeCardFile(path);
  }

  async function handleDeleteFolder(folderPath: string) {
    if (!activeProject) return;
    if (!confirm(`Delete folder "${folderPath}" and all its contents?`)) return;

    try {
      await deleteProjectFolder(activeProject, folderPath);
    } catch {
      // Folder might not exist on S3 yet
    }

    // Remove all files under this folder from the editor store
    const prefix = folderPath + "/";
    const toRemove = cardFiles
      .filter((f) => f.path.startsWith(prefix))
      .map((f) => f.path);
    for (const p of toRemove) {
      removeCardFile(p);
    }
  }

  function openNewCardDialog(parentPath: string) {
    setNewItemParent(parentPath);
    setNewCardName("");
    setShowNewCardDialog(true);
  }

  function openNewFolderDialog(parentPath: string) {
    setNewItemParent(parentPath);
    setNewFolderName("");
    setShowNewFolderDialog(true);
  }

  function renderNode(node: TreeNode, depth: number) {
    if (node.name === ".keep") return null;

    if (node.type === "folder") {
      const isExpanded = expandedFolders.has(node.path);
      // Filter out .keep children for display
      const visibleChildren = node.children.filter((c) => c.name !== ".keep");

      return (
        <div key={node.path}>
          <div
            onClick={() => toggleFolder(node.path)}
            onMouseEnter={() => setHoveredFile(node.path)}
            onMouseLeave={() => setHoveredFile(null)}
            className="flex items-center gap-1 px-2 py-1 cursor-pointer text-text-secondary hover:bg-border/20 hover:text-foreground transition-colors group"
            style={{ paddingLeft: `${depth * 12 + 8}px` }}
          >
            {isExpanded ? (
              <ChevronDown size={12} className="shrink-0" />
            ) : (
              <ChevronRight size={12} className="shrink-0" />
            )}
            {isExpanded ? (
              <FolderOpen size={14} className="text-yellow-600 shrink-0" />
            ) : (
              <Folder size={14} className="text-yellow-600 shrink-0" />
            )}
            <span className="flex-1 text-xs truncate">{node.name}</span>

            {hoveredFile === node.path && (
              <div className="flex items-center gap-0.5 shrink-0">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openNewCardDialog(node.path);
                  }}
                  className="p-0.5 rounded hover:bg-border/40 text-text-secondary/60 hover:text-foreground"
                  title="New card in folder"
                >
                  <Plus size={11} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openNewFolderDialog(node.path);
                  }}
                  className="p-0.5 rounded hover:bg-border/40 text-text-secondary/60 hover:text-foreground"
                  title="New subfolder"
                >
                  <FolderPlus size={11} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteFolder(node.path);
                  }}
                  className="p-0.5 rounded hover:bg-status-failed/10 text-text-secondary/60 hover:text-status-failed"
                  title="Delete folder"
                >
                  <Trash2 size={11} />
                </button>
              </div>
            )}
          </div>

          {isExpanded && (
            <div>
              {visibleChildren.map((child) => renderNode(child, depth + 1))}
              {visibleChildren.length === 0 && (
                <div
                  className="text-[10px] text-text-secondary/50 italic px-2 py-1"
                  style={{ paddingLeft: `${(depth + 1) * 12 + 8}px` }}
                >
                  Empty folder
                </div>
              )}
            </div>
          )}
        </div>
      );
    }

    // File node
    return (
      <div
        key={node.path}
        onClick={() => setActiveFilePath(node.path)}
        onMouseEnter={() => setHoveredFile(node.path)}
        onMouseLeave={() => setHoveredFile(null)}
        className={`flex items-center gap-2 px-2 py-1.5 cursor-pointer transition-colors group ${
          node.path === activeFilePath
            ? "bg-accent/15 text-foreground"
            : "text-text-secondary hover:bg-border/20 hover:text-foreground"
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <FileCode2 size={14} className="text-yellow-500 shrink-0" />
        <span className="flex-1 text-xs truncate">{node.name}</span>

        {cardFiles.find((f) => f.path === node.path)?.isDirty && (
          <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
        )}

        {hoveredFile === node.path && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDeleteFile(node.path);
            }}
            className="p-0.5 rounded hover:bg-status-failed/10 text-text-secondary/60 hover:text-status-failed shrink-0"
            title="Delete"
          >
            <Trash2 size={11} />
          </button>
        )}
      </div>
    );
  }

  return (
    <aside className="h-full bg-bg-secondary flex flex-col overflow-hidden">
      {/* Project selector */}
      <ProjectSelector />

      {/* Card files header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
          Card Files
        </h2>
        <div className="flex items-center gap-0.5">
          <button
            onClick={() => openNewFolderDialog("")}
            className="p-1 rounded hover:bg-border/40 text-text-secondary hover:text-foreground transition-colors"
            title="New folder"
          >
            <FolderPlus size={14} />
          </button>
          <button
            onClick={() => openNewCardDialog("")}
            className="p-1 rounded hover:bg-border/40 text-text-secondary hover:text-foreground transition-colors"
            title="New card file"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      {/* File tree */}
      <div className="flex-1 overflow-y-auto py-1">
        {!activeProject ? (
          <div className="px-3 py-8 text-center text-xs text-text-secondary">
            Select a project first.
          </div>
        ) : isLoadingFiles ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          </div>
        ) : tree.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-text-secondary">
            No card files yet.
            <br />
            Click + to create one.
          </div>
        ) : (
          tree.map((node) => renderNode(node, 0))
        )}
      </div>

      {/* New card dialog */}
      {showNewCardDialog && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80">
          <div className="bg-bg-secondary border border-border rounded-lg shadow-lg p-4 w-72">
            <h3 className="text-sm font-semibold text-foreground mb-1">
              New Card File
            </h3>
            {newItemParent && (
              <p className="text-[10px] text-text-secondary mb-2">
                in <span className="font-mono">{newItemParent}/</span>
              </p>
            )}
            <input
              type="text"
              value={newCardName}
              onChange={(e) => setNewCardName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreateCard()}
              placeholder="e.g. feature_scaler"
              className="w-full px-3 py-1.5 text-sm rounded border border-border bg-background text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              autoFocus
            />
            <p className="text-[10px] text-text-secondary mt-1.5">
              Use snake_case. The .py extension will be added automatically.
            </p>
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => {
                  setShowNewCardDialog(false);
                  setNewCardName("");
                }}
                className="px-3 py-1 text-xs rounded border border-border text-text-secondary hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateCard}
                disabled={!newCardName.trim()}
                className="px-3 py-1 text-xs rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New folder dialog */}
      {showNewFolderDialog && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80">
          <div className="bg-bg-secondary border border-border rounded-lg shadow-lg p-4 w-72">
            <h3 className="text-sm font-semibold text-foreground mb-1">
              New Folder
            </h3>
            {newItemParent && (
              <p className="text-[10px] text-text-secondary mb-2">
                in <span className="font-mono">{newItemParent}/</span>
              </p>
            )}
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreateFolder()}
              placeholder="e.g. training"
              className="w-full px-3 py-1.5 text-sm rounded border border-border bg-background text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
              autoFocus
            />
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => {
                  setShowNewFolderDialog(false);
                  setNewFolderName("");
                }}
                className="px-3 py-1 text-xs rounded border border-border text-text-secondary hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateFolder}
                disabled={!newFolderName.trim()}
                className="px-3 py-1 text-xs rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
