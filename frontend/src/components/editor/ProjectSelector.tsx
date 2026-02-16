"use client";

import { useState } from "react";
import { useWorkspaceStore } from "@/store/workspaceStore";
import { FolderOpen, ChevronDown, Plus, Trash2 } from "lucide-react";

interface ProjectSelectorProps {
  allowCreate?: boolean;
}

export function ProjectSelector({ allowCreate = true }: ProjectSelectorProps) {
  const projects = useWorkspaceStore((s) => s.projects);
  const activeProject = useWorkspaceStore((s) => s.activeProject);
  const switchProject = useWorkspaceStore((s) => s.switchProject);
  const createProject = useWorkspaceStore((s) => s.createProject);
  const deleteProject = useWorkspaceStore((s) => s.deleteProject);

  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [isSwitching, setIsSwitching] = useState(false);

  async function handleSwitch(name: string) {
    if (name === activeProject || isSwitching) return;
    setIsSwitching(true);
    try {
      await switchProject(name);
    } catch (err) {
      console.error("Failed to switch project:", err);
    } finally {
      setIsSwitching(false);
    }
  }

  async function handleDelete() {
    if (!activeProject) return;
    if (!confirm(`Delete project "${activeProject}" and all its cards and pipeline data? This cannot be undone.`)) return;
    try {
      await deleteProject(activeProject);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete project");
    }
  }

  async function handleCreate() {
    const name = newProjectName.trim();
    if (!name) return;
    try {
      await createProject(name);
      setNewProjectName("");
      setShowNewProject(false);
      await switchProject(name);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to create project");
    }
  }

  return (
    <div className="p-3 border-b border-border">
      <div className="flex items-center gap-1.5 mb-2">
        <FolderOpen size={13} className="text-accent shrink-0" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
          Project
        </span>
      </div>

      <div className="flex items-center gap-1">
        <div className="relative flex-1">
          <select
            value={activeProject || ""}
            onChange={(e) => handleSwitch(e.target.value)}
            disabled={isSwitching}
            className="w-full px-2.5 py-1.5 text-xs rounded border border-border bg-background text-foreground
              appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent
              disabled:opacity-50"
          >
            {!activeProject && (
              <option value="" disabled>
                Select a project...
              </option>
            )}
            {projects.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <ChevronDown
            size={12}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-text-secondary pointer-events-none"
          />
        </div>

        {activeProject && (
          <button
            onClick={handleDelete}
            className="p-1.5 rounded border border-border text-text-secondary
              hover:text-status-failed hover:border-status-failed/40 hover:bg-status-failed/10
              transition-colors shrink-0"
            title="Delete project"
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>

      {allowCreate && (
        <button
          onClick={() => setShowNewProject(true)}
          className="mt-2 w-full flex items-center justify-center gap-1 px-2 py-1 text-[10px]
            rounded border border-dashed border-border text-text-secondary
            hover:text-foreground hover:border-text-secondary/40 transition-colors"
        >
          <Plus size={11} />
          New Project
        </button>
      )}

      {allowCreate && showNewProject && (
        <div className="mt-2">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            placeholder="project name"
            className="w-full px-2 py-1 text-xs rounded border border-border bg-background text-foreground
              focus:outline-none focus:ring-1 focus:ring-accent"
            autoFocus
          />
          <div className="flex gap-1.5 mt-1.5">
            <button
              onClick={handleCreate}
              disabled={!newProjectName.trim()}
              className="flex-1 px-2 py-0.5 text-[10px] rounded bg-accent text-white hover:bg-accent/90
                disabled:opacity-50"
            >
              Create
            </button>
            <button
              onClick={() => {
                setShowNewProject(false);
                setNewProjectName("");
              }}
              className="flex-1 px-2 py-0.5 text-[10px] rounded border border-border text-text-secondary
                hover:text-foreground"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
