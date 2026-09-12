import React, { useState, useRef } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8008";

export default function FileUpload({ currentDoc, onDocUpdated }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [successInfo, setSuccessInfo] = useState(null);

  const fileInputRef = useRef(null);

  const stages = [
    "Uploading PDF...",
    "Extracting text...",
    "Cleaning text...",
    "Creating chunks...",
    "Generating embeddings...",
    "Building knowledge base...",
    "PDF Ready ?"
  ];

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith(".pdf")) {
        setFile(droppedFile);
        setErrorMsg("");
        setSuccessInfo(null);
      } else {
        setErrorMsg("Please select a valid PDF textbook file.");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (selected.name.toLowerCase().endsWith(".pdf")) {
        setFile(selected);
        setErrorMsg("");
        setSuccessInfo(null);
      } else {
        setErrorMsg("Please select a valid PDF textbook file.");
      }
    }
  };

  const processPDF = async () => {
    if (!file) {
      setErrorMsg("Please choose a PDF file first.");
      return;
    }

    setUploading(true);
    setErrorMsg("");
    setSuccessInfo(null);
    setStageIndex(0);

    const interval = setInterval(() => {
      setStageIndex((prev) => (prev < 5 ? prev + 1 : prev));
    }, 1100);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await axios.post(`${API_BASE}/upload-pdf`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      clearInterval(interval);
      setStageIndex(6); // PDF Ready ?
      setSuccessInfo(resp.data);
      onDocUpdated(resp.data);
      setFile(null);
    } catch (err) {
      clearInterval(interval);
      const detail = err.response?.data?.detail || err.message || "Failed to process PDF.";
      setErrorMsg(detail);
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="upload-container">
      <div className="section-header-box">
        <h2 className="section-title">Textbook Knowledge Source</h2>
        <p className="section-subtitle">
          Upload an educational textbook PDF. It becomes your active knowledge base for asking questions and practicing MCQs.
        </p>
      </div>

      <div
        className={`dropzone-card ${isDragging ? "dragging" : ""} ${uploading ? "busy" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="application/pdf"
          style={{ display: "none" }}
          disabled={uploading}
        />

        <div className="dropzone-content">
          <div className="upload-icon-circle">
            <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>

          <h3 className="dropzone-heading">
            {file ? file.name : "Upload your textbook PDF"}
          </h3>
          <p className="dropzone-desc">
            Drag and drop your PDF here, or click to browse
          </p>

          <div className="dropzone-actions" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              Choose PDF
            </button>

            {file && (
              <button
                type="button"
                className="btn btn-primary"
                onClick={processPDF}
                disabled={uploading}
              >
                {uploading ? "Processing..." : "Process PDF"}
              </button>
            )}
          </div>
        </div>
      </div>

      {uploading && (
        <div className="upload-progress-box">
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${((stageIndex + 1) / stages.length) * 100}%` }}
            ></div>
          </div>
          <div className="progress-stages-list">
            {stages.map((stg, idx) => (
              <div
                key={idx}
                className={`stage-item ${
                  idx < stageIndex ? "completed" : idx === stageIndex ? "current" : "pending"
                }`}
              >
                <span className="stage-icon">
                  {idx < stageIndex ? "?" : idx === stageIndex ? "?" : "?"}
                </span>
                <span>{stg}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="alert-box error">
          <span className="alert-icon">??</span>
          <span>{errorMsg}</span>
        </div>
      )}

      {(successInfo || (currentDoc && currentDoc.status === "ready")) && (
        <div className="ready-doc-card">
          <div className="ready-header">
            <span className="check-badge">?</span>
            <span className="ready-title">PDF Ready</span>
          </div>

          <div className="ready-grid">
            <div className="ready-item">
              <span className="ready-label">Current Document</span>
              <span className="ready-value">
                {successInfo?.filename || currentDoc?.filename}
              </span>
            </div>
            <div className="ready-item">
              <span className="ready-label">Pages</span>
              <span className="ready-value">
                {successInfo?.pages || currentDoc?.pages}
              </span>
            </div>
            <div className="ready-item">
              <span className="ready-label">Chunks</span>
              <span className="ready-value">
                {successInfo?.chunks || currentDoc?.chunks}
              </span>
            </div>
            <div className="ready-item">
              <span className="ready-label">Status</span>
              <span className="status-badge ready">Ready</span>
            </div>
          </div>

          <p className="ready-note">
            Note: The newly uploaded PDF is now your current active knowledge source.
          </p>
        </div>
      )}
    </section>
  );
}
