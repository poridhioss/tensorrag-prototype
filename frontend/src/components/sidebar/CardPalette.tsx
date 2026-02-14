"use client";

import { usePipelineStore } from "@/store/pipelineStore";
import type { CardSchema } from "@/lib/types";

const CATEGORY_ORDER = ["data", "model", "training", "evaluation", "inference"] as const;
const CATEGORY_LABELS: Record<string, string> = {
  data: "Data",
  model: "Model",
  training: "Training Steps",
  evaluation: "Evaluation",
  inference: "Inference",
};

export function CardPalette() {
  const cardSchemas = usePipelineStore((s) => s.cardSchemas);

  const grouped = CATEGORY_ORDER.reduce(
    (acc, cat) => {
      acc[cat] = cardSchemas.filter((c) => c.category === cat);
      return acc;
    },
    {} as Record<string, CardSchema[]>
  );

  const onDragStart = (e: React.DragEvent, cardType: string) => {
    e.dataTransfer.setData("application/tensorrag-card", cardType);
    e.dataTransfer.effectAllowed = "move";
  };

  return (
    <aside className="w-56 border-r border-border bg-bg-secondary shrink-0 overflow-y-auto">
      <div className="p-3">
        <h2 className="text-xs font-semibold uppercase text-text-secondary mb-3">
          Cards
        </h2>
        {CATEGORY_ORDER.map((cat) => {
          const cards = grouped[cat] || [];
          if (cards.length === 0) return null;

          // Training steps are shown as a subcategory under Model
          const isTraining = cat === "training";

          return (
            <div key={cat} className={isTraining ? "mb-4 ml-3 border-l-2 border-border/50 pl-3 mt-2" : "mb-4"}>
              <h3 className={`text-[10px] font-semibold uppercase mb-1.5 ${isTraining ? "text-text-secondary/70" : "text-text-secondary"}`}>
                {CATEGORY_LABELS[cat]}
              </h3>
              <div className="space-y-1.5">
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
            </div>
          );
        })}
        {cardSchemas.length === 0 && (
          <div className="text-xs text-text-secondary animate-pulse">
            Loading cards...
          </div>
        )}
      </div>
    </aside>
  );
}
