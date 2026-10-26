import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8008";

export default function MCQPractice({ currentDoc }) {
  const [topic, setTopic] = useState("");
  const [mcqs, setMcqs] = useState([]);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  const sampleTopics = [
    "Chromosomal Theory of Inheritance",
    "Simple Harmonic Motion",
    "Energy in SHM",
    "Henrys Law",
    "Resonance"
  ];

  const handleGenerateMCQs = async (chosenTopic) => {
    const t = (chosenTopic || topic).trim();
    if (!t) return;

    if (!currentDoc || currentDoc.status !== "ready") {
      setErrorMsg("Please upload a textbook PDF first.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setMcqs([]);
    setSelectedAnswers({});
    setIsSubmitted(false);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000);

    try {
      const resp = await axios.post(`${API_BASE}/generate-mcq`, { topic: t });
      if (resp.data && resp.data.mcqs && resp.data.mcqs.length > 0) {
        setMcqs(resp.data.mcqs);
      } else {
        throw new Error("Could not generate MCQs for this topic from the uploaded textbook.");
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || "Failed to generate MCQs.";
      setErrorMsg(detail);
    } finally {
      setLoading(false);
      clearInterval(timerRef.current);
    }
  };

  const handleSelectOption = (questionId, optionKey) => {
    if (isSubmitted) return;
    setSelectedAnswers((prev) => ({
      ...prev,
      [questionId]: optionKey,
    }));
  };

  const handleCheckAnswers = () => {
    if (Object.keys(selectedAnswers).length === 0) {
      setErrorMsg("Please select at least one option before checking answers.");
      return;
    }
    setErrorMsg("");
    setIsSubmitted(true);
  };

  const calculateScore = () => {
    let score = 0;
    mcqs.forEach((m) => {
      const correct = m.correct_answer || m.answer;
      if (selectedAnswers[m.id] === correct) {
        score += 1;
      }
    });
    return score;
  };

  const handleReset = () => {
    setSelectedAnswers({});
    setIsSubmitted(false);
  };

  return (
    <div className="tab-pane">
      <div className="section-header-box">
        <h2 className="section-title">Practice MCQs</h2>
        <p className="section-subtitle">
          Test your conceptual understanding with 5 exam-style multiple choice questions generated directly from your textbook.
        </p>
      </div>

      <div className="topic-card card">
        <label className="input-label" htmlFor="topic-input">
          Enter topic:
        </label>
        <div className="topic-input-group">
          <input
            id="topic-input"
            type="text"
            className="input-text"
            placeholder="e.g. Chromosomal Theory of Inheritance"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleGenerateMCQs();
            }}
          />
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => handleGenerateMCQs()}
            disabled={loading || !topic.trim()}
          >
            {loading ? (
              <>
                <span className="spinner"></span> Generating...
              </>
            ) : (
              "Generate 5 MCQs"
            )}
          </button>
        </div>

        <div className="suggestion-chips">
          <span className="chips-label">Popular topics:</span>
          {sampleTopics.map((st, idx) => (
            <button
              key={idx}
              type="button"
              className="chip-btn"
              onClick={() => {
                setTopic(st);
                handleGenerateMCQs(st);
              }}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {errorMsg && (
        <div className="alert-box error">
          <span className="alert-icon">!</span>
          <span>{errorMsg}</span>
        </div>
      )}

      {loading && (
        <div className="loading-card card">
          <span className="spinner large"></span>
          <div className="loading-text">
            <h3>Generating 5 Exam-Style MCQs...</h3>
            <p>Extracting relevant concepts from the active textbook and drafting exam-quality questions and explanations.</p>
          </div>
        </div>
      )}

      {mcqs.length > 0 && (
        <div className="quiz-container">
          <div className="quiz-header">
            <div>
              <h3 className="quiz-title">Quiz: {topic}</h3>
              <p className="quiz-subtitle">5 Exam Questions - 4 Options Each</p>
            </div>

            {isSubmitted && (
              <div className="score-badge">
                <span className="score-label">Score:</span>
                <span className="score-value">
                  {calculateScore()} / {mcqs.length}
                </span>
                <span className="score-pct">
                  ({Math.round((calculateScore() / mcqs.length) * 100)}%)
                </span>
              </div>
            )}
          </div>

          <div className="mcq-list">
            {mcqs.map((mcq, qIdx) => {
              const selectedKey = selectedAnswers[mcq.id];
              const correctKey = mcq.correct_answer || mcq.answer;
              const isCorrect = selectedKey === correctKey;

              return (
                <div
                  key={mcq.id || qIdx}
                  className={`mcq-card ${
                    isSubmitted
                      ? isCorrect
                        ? "graded-correct"
                        : "graded-incorrect"
                      : ""
                  }`}
                >
                  <div className="mcq-card-header">
                    <span className="q-number">Q{qIdx + 1}</span>
                    <h4 className="q-text">{mcq.question}</h4>
                    <span className="source-page-pill">Source: Page {mcq.page}</span>
                  </div>

                  <div className="mcq-options-grid">
                    {Object.entries(mcq.options || {}).map(([optKey, optText]) => {
                      const isSelected = selectedKey === optKey;
                      const isOptionCorrect = optKey === correctKey;

                      let optClass = "mcq-option-btn";
                      if (isSelected) optClass += " selected";

                      if (isSubmitted) {
                        if (isOptionCorrect) {
                          optClass += " correct-ans";
                        } else if (isSelected && !isOptionCorrect) {
                          optClass += " wrong-ans";
                        }
                      }

                      return (
                        <button
                          key={optKey}
                          type="button"
                          className={optClass}
                          onClick={() => handleSelectOption(mcq.id, optKey)}
                          disabled={isSubmitted}
                        >
                          <span className="option-marker">{optKey}</span>
                          <span className="option-text">{optText}</span>
                          {isSubmitted && isOptionCorrect && (
                            <span className="indicator-icon correct">{"\u2713"}</span>
                          )}
                          {isSubmitted && isSelected && !isOptionCorrect && (
                            <span className="indicator-icon wrong">{"\u00d7"}</span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {isSubmitted && (
                    <div className={`explanation-card ${isCorrect ? "success" : "warning"}`}>
                      {isCorrect ? (
                        <>
                          <div className="eval-status-row">
                            <span className="eval-badge correct">{"\u2713"} Correct</span>
                            <span className="student-ans-note">
                              Your Answer: <strong>Option {selectedKey}</strong>
                            </span>
                          </div>
                          <p className="explanation-text">
                            <strong>Explanation:</strong> {mcq.explanation}
                          </p>
                          <span className="eval-page-ref">Source: Page {mcq.page}</span>
                        </>
                      ) : (
                        <>
                          <div className="eval-status-row">
                            <span className="eval-badge incorrect">{"\u00d7"} Incorrect</span>
                            <span className="student-ans-note">
                              Your Answer: <strong>Option {selectedKey || "None"}</strong> | Correct Answer: <strong>Option {correctKey}</strong>
                            </span>
                          </div>

                          {selectedKey && mcq.distractor_explanations?.[selectedKey] && (
                            <p className="explanation-text mistake-reason">
                              <strong>Why your answer is incorrect:</strong>{" "}
                              {mcq.distractor_explanations[selectedKey]}
                            </p>
                          )}

                          <p className="explanation-text correct-reason">
                            <strong>Why {correctKey} is correct:</strong> {mcq.explanation}
                          </p>

                          <span className="eval-page-ref">Source: Page {mcq.page}</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="quiz-footer">
            {!isSubmitted ? (
              <button
                type="button"
                className="btn btn-primary large"
                onClick={handleCheckAnswers}
                disabled={Object.keys(selectedAnswers).length === 0}
              >
                Check Answers
              </button>
            ) : (
              <div className="quiz-result-actions">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleReset}
                >
                  Retry This Quiz
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => {
                    setMcqs([]);
                    setSelectedAnswers({});
                    setIsSubmitted(false);
                    setTopic("");
                  }}
                >
                  Try Another Topic
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
