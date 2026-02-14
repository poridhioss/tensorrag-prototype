"use client";

import { useEffect, useRef } from "react";
import type { WSNodeStatusMessage, WSMessage } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function useWebSocket(
  pipelineId: string | null,
  onNodeStatus: (msg: WSNodeStatusMessage) => void
) {
  const wsRef = useRef<WebSocket | null>(null);
  const callbackRef = useRef(onNodeStatus);
  callbackRef.current = onNodeStatus;

  useEffect(() => {
    if (!pipelineId) return;

    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_URL}/ws/pipeline/${pipelineId}`);

    ws.onmessage = (event) => {
      try {
        const msg: WSNodeStatusMessage = JSON.parse(event.data);
        if (msg.type === "node_status") {
          callbackRef.current(msg);
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    wsRef.current = ws;

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [pipelineId]);
}

/** Connect a WebSocket imperatively and return it. Resolves once open. */
export function connectWebSocket(
  pipelineId: string,
  onMessage: (msg: WSMessage) => void
): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${WS_URL}/ws/pipeline/${pipelineId}`);

    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("WebSocket connection failed"));

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        onMessage(msg);
      } catch {
        // Ignore
      }
    };
  });
}
