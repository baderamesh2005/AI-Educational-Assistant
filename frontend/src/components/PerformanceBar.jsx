import React from "react";

export default function PerformanceBar({ timing }) {
  if (!timing) return null;

  const embeddingMs = Math.round((timing.embedding_sec || 0) * 1000);
  const retrievalMs = Math.round((timing.retrieval_sec || 0) * 1000);
  const llmSec = timing.llm_sec != null ? Number(timing.llm_sec).toFixed(2) : "0.00";
  const totalSec = timing.total_sec != null ? Number(timing.total_sec).toFixed(2) : "0.00";

  return (
    <div className="perf-bar">
      <div className="perf-title">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
        <span>Latency Breakdown:</span>
      </div>
      <div className="perf-chips">
        <span className="perf-chip">
          <span className="perf-label">Embedding:</span> {embeddingMs}ms
        </span>
        <span className="perf-chip">
          <span className="perf-label">FAISS Retrieval:</span> {retrievalMs}ms
        </span>
        <span className="perf-chip">
          <span className="perf-label">Ollama Generation:</span> {llmSec}s
        </span>
        <span className="perf-chip total">
          <span className="perf-label">Total:</span> {totalSec}s
        </span>
      </div>
    </div>
  );
}
