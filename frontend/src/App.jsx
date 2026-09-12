import React, { useState, useEffect } from "react";
import axios from "axios";
import Header from "./components/Header.jsx";
import FileUpload from "./components/FileUpload.jsx";
import AskAI from "./components/AskAI.jsx";
import MCQPractice from "./components/MCQPractice.jsx";

const API_BASE = "http://127.0.0.1:8008";

export default function App() {
  const [activeTab, setActiveTab] = useState("ask"); // "ask" | "mcq"
  const [currentDoc, setCurrentDoc] = useState(null);
  const [health, setHealth] = useState(null);

  const fetchStatus = async () => {
    try {
      const [docRes, healthRes] = await Promise.all([
        axios.get(`${API_BASE}/current-document`),
        axios.get(`${API_BASE}/health`),
      ]);
      setCurrentDoc(docRes.data);
      setHealth(healthRes.data);
    } catch (err) {
      console.error("Error fetching system status:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 25000);
    return () => clearInterval(timer);
  }, []);

  const handleDocUpdated = (newDocInfo) => {
    setCurrentDoc({
      filename: newDocInfo.filename,
      pages: newDocInfo.pages,
      chunks: newDocInfo.chunks,
      status: "ready",
    });
  };

  return (
    <div className="app-layout">
      <Header currentDoc={currentDoc} health={health} />

      <main className="main-container">
        {/* PDF Upload Section */}
        <FileUpload currentDoc={currentDoc} onDocUpdated={handleDocUpdated} />

        {/* Tab Navigation */}
        <div className="tabs-container">
          <div className="tabs-nav" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "ask"}
              className={`tab-btn ${activeTab === "ask" ? "active" : ""}`}
              onClick={() => setActiveTab("ask")}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span>Ask AI</span>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "mcq"}
              className={`tab-btn ${activeTab === "mcq" ? "active" : ""}`}
              onClick={() => setActiveTab("mcq")}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="m9 12 2 2 4-4" />
              </svg>
              <span>Practice MCQs</span>
            </button>
          </div>
        </div>

        {/* Active Tab View */}
        <div className="tab-content-area">
          {activeTab === "ask" && <AskAI currentDoc={currentDoc} />}
          {activeTab === "mcq" && <MCQPractice currentDoc={currentDoc} />}
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-inner">
          <p>AI Educational Assistant</p>
        </div>
      </footer>
    </div>
  );
}
