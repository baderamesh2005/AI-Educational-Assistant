import React, { useState, useRef } from "react";

const API_BASE = "http://127.0.0.1:8008";

export default function AskAI({ currentDoc }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [primaryPage, setPrimaryPage] = useState(null);
  const [confidencePct, setConfidencePct] = useState(null);
  const [sourceMatchPct, setSourceMatchPct] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");

  const abortControllerRef = useRef(null);

  const sampleQuestions = [
    "Explain the chromosomal theory of inheritance.",
    "What is simple harmonic motion?",
    "Explain resonance and state its condition.",
    "State Henrys Law and its formula."
  ];

  const handleAsk = async (qText) => {
    const q = (qText || question).trim();
    if (!q) return;

    if (!currentDoc || currentDoc.status !== "ready") {
      setErrorMsg("Please upload a textbook PDF first.");
      return;
    }

    setErrorMsg("");
    setLoading(true);
    setAnswer("");
    setPrimaryPage(null);
    setConfidencePct(null);
    setSourceMatchPct(null);
    setSources([]);
    setSubmittedQuestion(q);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const dataStr = trimmed.slice(6);
            try {
              const payload = JSON.parse(dataStr);
              if (payload.type === "meta") {
                if (payload.primary_page != null) setPrimaryPage(payload.primary_page);
                if (payload.confidence_pct != null) setConfidencePct(payload.confidence_pct);
                if (payload.source_match_pct != null) setSourceMatchPct(payload.source_match_pct);
                if (payload.sources) setSources(payload.sources);
              } else if (payload.type === "token") {
                setAnswer((prev) => prev + payload.token);
              }
            } catch (e) {
              // Ignore partial JSON
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setErrorMsg(err.message || "Failed to generate answer. Check if Ollama is running.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-pane">
      <div className="section-header-box">
        <h2 className="section-title">Ask AI</h2>
        <p className="section-subtitle">
          Ask questions grounded strictly in your uploaded textbook. Zero hallucinations.
        </p>
      </div>

      <div className="query-box card">
        <label className="input-label" htmlFor="qa-input">
          Ask a question about your uploaded textbook:
        </label>
        <div className="textarea-wrapper">
          <textarea
            id="qa-input"
            className="query-textarea"
            rows="3"
            placeholder="e.g. Explain the chromosomal theory of inheritance."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                handleAsk();
              }
            }}
          />
        </div>

        <div className="query-footer">
          <div className="suggestion-chips">
            <span className="chips-label">Try:</span>
            {sampleQuestions.map((sq, idx) => (
              <button
                key={idx}
                type="button"
                className="chip-btn"
                onClick={() => {
                  setQuestion(sq);
                  handleAsk(sq);
                }}
              >
                {sq}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="btn btn-primary submit-btn"
            onClick={() => handleAsk()}
            disabled={loading || !question.trim()}
          >
            {loading ? (
              <>
                <span className="spinner"></span> AI is thinking...
              </>
            ) : (
              "Ask AI"
            )}
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="alert-box error">
          <span className="alert-icon">??</span>
          <span>{errorMsg}</span>
        </div>
      )}

      {(submittedQuestion || answer || loading) && (
        <div className="answer-section card">
          <div className="answer-header">
            <div>
              <h3 className="answer-title">AI Answer</h3>
              <p className="answer-query-echo">Q: {submittedQuestion}</p>
              {primaryPage && (
                <div className="answer-source-line">
                  Source: <strong>Textbook Page {primaryPage}</strong>
                </div>
              )}
            </div>
          </div>

          <div className="answer-divider"></div>

          <div className="answer-body">
            {answer ? (
              <div className="markdown-content">
                {answer}
                {loading && <span className="typing-cursor"></span>}
              </div>
            ) : loading ? (
              <div className="loading-state">
                <span className="spinner large"></span>
                <span>Retrieving textbook context and drafting exam-ready answer...</span>
              </div>
            ) : null}
          </div>

          {confidencePct != null && !loading && (
            <>
              <div className="answer-divider"></div>
              <div className="confidence-section">
                <div className="confidence-metric">
                  <div className="metric-header">
                    <span className="metric-title">Answer Confidence:</span>
                    <span className="metric-value">{confidencePct}%</span>
                  </div>
                  <div className="confidence-bar-track">
                    <div
                      className="confidence-bar-fill"
                      style={{ width: `${confidencePct}%` }}
                    ></div>
                  </div>
                </div>

                {sourceMatchPct != null && (
                  <div className="confidence-metric">
                    <div className="metric-header">
                      <span className="metric-title">Source Match:</span>
                      <span className="metric-value">{sourceMatchPct}%</span>
                    </div>
                    <div className="confidence-bar-track">
                      <div
                        className="confidence-bar-fill match"
                        style={{ width: `${sourceMatchPct}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {sources && sources.length > 0 && !loading && (
            <>
              <div className="answer-divider"></div>
              <div className="sources-container">
                <h4 className="sources-heading">
                  Referenced Textbook Sources
                </h4>

                <div className="sources-grid">
                  {sources.map((src, idx) => (
                    <div key={idx} className="source-card">
                      <div className="source-card-header">
                        <span className="page-tag">Page {src.page}</span>
                        {idx === 0 && <span className="primary-tag">Primary Source</span>}
                      </div>
                      <p className="source-snippet">"{src.text}"</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
