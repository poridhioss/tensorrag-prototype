import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  listProjects,
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  activateProject,
} from "@/lib/api";
import { usePipelineStore } from "./pipelineStore";

interface WorkspaceState {
  projects: string[];
  activeProject: string | null;
  isLoadingProjects: boolean;
  /** Maps card_type → file path (e.g. "data_load" → "data/data_load.py") */
  cardPathMap: Record<string, string>;

  loadProjects: () => Promise<void>;
  setActiveProject: (name: string | null) => void;
  setCardPathMap: (map: Record<string, string>) => void;
  createProject: (name: string) => Promise<void>;
  deleteProject: (name: string) => Promise<void>;
  switchProject: (name: string) => Promise<void>;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
      projects: [],
      activeProject: null,
      isLoadingProjects: false,
      cardPathMap: {},

      loadProjects: async () => {
        set({ isLoadingProjects: true });
        try {
          const projects = await listProjects();
          set({ projects });
        } catch {
          // Backend not reachable
        } finally {
          set({ isLoadingProjects: false });
        }
      },

      setActiveProject: (name) => set({ activeProject: name }),

      setCardPathMap: (map) => set({ cardPathMap: map }),

      createProject: async (name: string) => {
        await apiCreateProject(name);
        const projects = await listProjects();
        set({ projects });
      },

      deleteProject: async (name: string) => {
        await apiDeleteProject(name);
        const projects = await listProjects();
        const active = get().activeProject;
        set({
          projects,
          activeProject: active === name ? null : active,
        });
      },

      switchProject: async (name: string) => {
        const pipelineStore = usePipelineStore.getState();

        // Save current pipeline state if there's an active project
        const currentProject = get().activeProject;
        if (currentProject) {
          await pipelineStore.savePipelineToServer(currentProject);
        }

        // Activate the new project (registers its cards in backend)
        const result = await activateProject(name);
        pipelineStore.setCardSchemas(result.cards);

        // Build card_type → file path mapping
        const pathMap: Record<string, string> = {};
        for (const entry of result.registered) {
          pathMap[entry.card_type] = entry.path;
        }

        // Load the new project's pipeline state
        await pipelineStore.loadPipelineFromServer(name);

        set({ activeProject: name, cardPathMap: pathMap });
      },
    }),
    {
      name: "tensorrag-workspace",
      partialize: (state) => ({
        activeProject: state.activeProject,
      }),
    }
  )
);
