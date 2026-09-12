import React from "react";

export default function Header({ currentDoc, health }) {
  const isOllamaOnline = health?.ollama?.online;
  const modelName = health?.ollama?.model || "llama3.2:3b";

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <svg className="brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              <path d="M12 6v6" />
              <path d="M9 9h6" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">AI Educational Assistant</h1>
            <p className="brand-subtitle">Learn from your own textbook using AI</p>
          </div>
        </div>

        <div className="header-status-area">
          {currentDoc && currentDoc.status === "ready" ? (
            <div className="active-doc-badge">
              <div className="active-doc-details">
                <span className="active-doc-name" title={currentDoc.filename}>
                  {currentDoc.filename}
                </span>
                <span className="active-doc-meta">
                  {currentDoc.pages} Pages · {currentDoc.chunks} Chunks
                </span>
              </div>
              <span className="badge-pill active">Active Source</span>
            </div>
          ) : (
            <div className="no-doc-badge">
              <span className="status-dot warning"></span>
              <span>No PDF Active</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

