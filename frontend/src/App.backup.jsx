import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState("detection");
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [history, setHistory] = useState(() => {
    try {
      const savedHistory = localStorage.getItem("fakeNewsHistory");
      return savedHistory ? JSON.parse(savedHistory) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem(
      "fakeNewsHistory",
      JSON.stringify(history)
    );
  }, [history]);

  const totalScans = history.length;

  const fakeDetected = history.filter((item) =>
    (item.prediction || "").toUpperCase().includes("FAKE")
  ).length;

  const selectMode = (selectedMode) => {
    setMode(selectedMode);
    setResult(null);
    setText("");
    setImage(null);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview("");
  };

  const handleImage = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const saveHistory = (data, analysisMode) => {
    const historyItem = {
      id: Date.now(),
      prediction: data.prediction,
      confidence: Number(data.confidence || 0),
      fake_probability: Number(
        data.fake_probability || 0
      ),
      real_probability: Number(
        data.real_probability || 0
      ),
      analysisMode,
      input:
        analysisMode === "text"
          ? text
          : image?.name || "News Image",
      ocr_text: data.ocr_text || "",
      date: new Date().toLocaleString(),
    };

    setHistory((previousHistory) => [
      historyItem,
      ...previousHistory,
    ]);
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      alert("Please enter news text.");
      return;
    }

    try {
      setLoading(true);
      setResult(null);

      const formData = new URLSearchParams();
      formData.append("text", text);

      const response = await fetch(
        `${API_URL}/predict-text`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/x-www-form-urlencoded",
          },
          body: formData.toString(),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Text analysis failed"
        );
      }

      setResult(data);
      saveHistory(data, "text");
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      alert("Please select an image.");
      return;
    }

    try {
      setLoading(true);
      setResult(null);

      const formData = new FormData();
      formData.append("file", image);
      formData.append("text", "");

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Image analysis failed"
        );
      }

      setResult(data);
      saveHistory(data, "image");
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const resetInput = () => {
    setText("");
    setImage(null);
    setResult(null);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview("");
  };

  const deleteHistoryItem = (id) => {
    setHistory((previousHistory) =>
      previousHistory.filter((item) => item.id !== id)
    );
  };

  const clearHistory = () => {
    if (history.length === 0) return;

    const confirmed = window.confirm(
      "Clear all detection history?"
    );

    if (confirmed) {
      setHistory([]);
    }
  };

  const openHistoryResult = (item) => {
    setResult(item);
    setMode(item.analysisMode);
    setActivePage("detection");
  };

  const fakeProbability = Number(
    result?.fake_probability || 0
  );

  const realProbability = Number(
    result?.real_probability || 0
  );

  const confidence = Number(result?.confidence || 0);

  const isFake = (result?.prediction || "")
    .toUpperCase()
    .includes("FAKE");

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">FN</div>

          <div>
            <h1>FakeNews AI</h1>
            <p>DETECTION SYSTEM</p>
          </div>
        </div>

        <div className="model-card">
          <div className="model-icon">AI</div>

          <div>
            <h3>AI Detection</h3>
            <p>Fake News Analysis</p>
          </div>
        </div>

        <nav className="navigation">
          <button
            className={`nav-item ${
              activePage === "detection" ? "active" : ""
            }`}
            onClick={() => setActivePage("detection")}
          >
            <span>⌂</span>
            Detection
          </button>

          <button
            className={`nav-item ${
              activePage === "history" ? "active" : ""
            }`}
            onClick={() => setActivePage("history")}
          >
            <span>↻</span>
            History
          </button>

          <button
            className={`nav-item ${
              activePage === "about" ? "active" : ""
            }`}
            onClick={() => setActivePage("about")}
          >
            <span>ⓘ</span>
            About Project
          </button>
        </nav>

        <div className="metrics">
          <p className="section-label">
            DETECTION METRICS
          </p>

          <div className="metric-grid">
            <div className="metric-box">
              <strong>{totalScans}</strong>
              <span>TOTAL SCANS</span>
            </div>

            <div className="metric-box">
              <strong>{fakeDetected}</strong>
              <span>FAKE DETECTED</span>
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>

          <div>
            <p>AI MODEL</p>
            <strong>ONLINE</strong>
          </div>
        </div>
      </aside>

      <main className="workspace">
        {activePage === "detection" && (
          <>
            <header className="topbar">
              <div>
                <h2>Fake News Detection</h2>
                <p>Select text or image analysis</p>
              </div>

              <div className="api-status">
                <span className="status-dot"></span>

                <div>
                  <small>Prediction API</small>
                  <strong>Online</strong>
                </div>
              </div>
            </header>

            <section className="analysis-area">
              <div className="mode-title">
                <h3>Select Analysis Type</h3>
              </div>

              <div className="mode-selector">
                <button
                  className={`mode-card ${
                    mode === "text" ? "selected" : ""
                  }`}
                  onClick={() => selectMode("text")}
                >
                  <div className="mode-icon">T</div>

                  <div>
                    <h3>Text Analysis</h3>
                    <p>
                      Detect fake or real news from text
                    </p>
                  </div>
                </button>

                <button
                  className={`mode-card ${
                    mode === "image" ? "selected" : ""
                  }`}
                  onClick={() => selectMode("image")}
                >
                  <div className="mode-icon">IMG</div>

                  <div>
                    <h3>Image Analysis</h3>
                    <p>
                      Detect fake or real news image
                    </p>
                  </div>
                </button>
              </div>

              <div className="input-panel">
                {mode === "text" ? (
                  <div className="text-analysis">
                    <div className="panel-heading">
                      <div>
                        <span className="panel-number">
                          01
                        </span>
                        <h3>Enter News Text</h3>
                      </div>

                      <span>{text.length} / 5000</span>
                    </div>

                    <textarea
                      value={text}
                      maxLength={5000}
                      onChange={(event) =>
                        setText(event.target.value)
                      }
                      placeholder="Enter news text here..."
                    />

                    <div className="action-row">
                      <button
                        className="reset-btn"
                        onClick={resetInput}
                      >
                        Clear
                      </button>

                      <button
                        className="analyze-btn"
                        onClick={analyzeText}
                        disabled={loading}
                      >
                        {loading
                          ? "Analyzing..."
                          : "Analyze Text"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="image-analysis">
                    <div className="panel-heading">
                      <div>
                        <span className="panel-number">
                          01
                        </span>
                        <h3>Select News Image</h3>
                      </div>
                    </div>

                    <label className="upload-box">
                      <input
                        type="file"
                        accept="image/png,image/jpeg,image/jpg"
                        onChange={handleImage}
                      />

                      {preview ? (
                        <img
                          className="image-preview"
                          src={preview}
                          alt="Selected"
                        />
                      ) : (
                        <div className="upload-content">
                          <div className="upload-icon">
                            ↑
                          </div>

                          <h3>Select Image</h3>

                          <p>
                            Click to choose JPG, JPEG or
                            PNG
                          </p>
                        </div>
                      )}
                    </label>

                    {image && (
                      <div className="file-info">
                        <span>✓</span>
                        {image.name}
                      </div>
                    )}

                    <div className="action-row">
                      <button
                        className="reset-btn"
                        onClick={resetInput}
                      >
                        Clear
                      </button>

                      <button
                        className="analyze-btn"
                        onClick={analyzeImage}
                        disabled={loading}
                      >
                        {loading
                          ? "Analyzing..."
                          : "Analyze Image"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </>
        )}

        {activePage === "history" && (
          <section className="page-content">
            <div className="page-header">
              <div>
                <span className="page-label">
                  ANALYSIS RECORDS
                </span>
                <h1>Detection History</h1>
                <p>
                  Previous fake news detection results
                </p>
              </div>

              {history.length > 0 && (
                <button
                  className="clear-history-btn"
                  onClick={clearHistory}
                >
                  Clear History
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="history-empty">
                <div className="history-icon">↻</div>

                <h2>No Detection History</h2>

                <p>
                  Your text and image analysis results
                  will appear here.
                </p>

                <button
                  className="analyze-btn"
                  onClick={() =>
                    setActivePage("detection")
                  }
                >
                  Start Detection
                </button>
              </div>
            ) : (
              <div className="history-list">
                {history.map((item) => {
                  const itemIsFake = (
                    item.prediction || ""
                  )
                    .toUpperCase()
                    .includes("FAKE");

                  return (
                    <div
                      className="history-card"
                      key={item.id}
                    >
                      <div
                        className={`history-status ${
                          itemIsFake
                            ? "history-fake"
                            : "history-real"
                        }`}
                      >
                        {itemIsFake ? "F" : "R"}
                      </div>

                      <div className="history-details">
                        <div className="history-top">
                          <h3>{item.prediction}</h3>

                          <span>
                            {item.analysisMode === "text"
                              ? "TEXT"
                              : "IMAGE"}
                          </span>
                        </div>

                        <p className="history-input">
                          {item.input}
                        </p>

                        <div className="history-meta">
                          <span>
                            Confidence:{" "}
                            {Number(
                              item.confidence || 0
                            ).toFixed(1)}
                            %
                          </span>

                          <span>{item.date}</span>
                        </div>
                      </div>

                      <div className="history-actions">
                        <button
                          className="view-history-btn"
                          onClick={() =>
                            openHistoryResult(item)
                          }
                        >
                          View
                        </button>

                        <button
                          className="delete-history-btn"
                          onClick={() =>
                            deleteHistoryItem(item.id)
                          }
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

        {activePage === "about" && (
          <section className="page-content">
            <div className="page-header">
              <div>
                <span className="page-label">
                  PROJECT INFORMATION
                </span>

                <h1>About Project</h1>

                <p>
                  Multimodal Fake News Detection System
                </p>
              </div>
            </div>

            <div className="project-intro">
              <div className="project-logo">FN</div>

              <div>
                <h2>FakeNews AI</h2>

                <p>
                  An artificial intelligence based fake
                  news detection application designed to
                  analyze textual and visual news
                  content.
                </p>
              </div>
            </div>

            <div className="about-grid">
              <div className="about-card">
                <span>TEXT ANALYSIS</span>
                <h2>BERT</h2>
                <p>
                  The text detection model uses BERT
                  based 768 dimensional semantic
                  features to classify fake and real
                  news text.
                </p>
              </div>

              <div className="about-card">
                <span>IMAGE ANALYSIS</span>
                <h2>ResNet18</h2>
                <p>
                  ResNet18 extracts visual features from
                  uploaded news images for image based
                  fake news analysis.
                </p>
              </div>

              <div className="about-card">
                <span>BACKEND API</span>
                <h2>FastAPI</h2>
                <p>
                  FastAPI connects the trained machine
                  learning models with the frontend and
                  provides prediction endpoints.
                </p>
              </div>

              <div className="about-card">
                <span>USER INTERFACE</span>
                <h2>React</h2>
                <p>
                  React provides a responsive interface
                  for selecting text or image analysis
                  and viewing AI prediction results.
                </p>
              </div>

              <div className="about-card">
                <span>OCR SYSTEM</span>
                <h2>Text Extraction</h2>
                <p>
                  OCR extracts visible text from
                  uploaded news images for additional
                  content analysis.
                </p>
              </div>

              <div className="about-card">
                <span>DOMAIN</span>
                <h2>Artificial Intelligence</h2>
                <p>
                  The project combines Machine Learning,
                  Natural Language Processing and
                  Computer Vision.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      <aside className="result-panel">
        <div className="result-header">
          <h2>Detection Result</h2>
          <p>AI PREDICTION OUTPUT</p>
        </div>

        {!result && !loading && (
          <div className="empty-result">
            <div className="empty-icon">×</div>
            <h3>No Active Analysis</h3>
            <p>
              Select text or image and start analysis.
            </p>
          </div>
        )}

        {loading && (
          <div className="loading-result">
            <div className="loader"></div>
            <h3>Analyzing...</h3>
            <p>AI model is processing your input.</p>
          </div>
        )}

        {result && !loading && (
          <div className="result-content">
            <div
              className={`prediction-card ${
                isFake ? "fake-result" : "real-result"
              }`}
            >
              <span>MODEL PREDICTION</span>
              <h2>{result.prediction}</h2>
              <strong>{confidence.toFixed(1)}%</strong>
              <p>Confidence Score</p>
            </div>

            <div className="probability-section">
              <div className="probability-item">
                <div className="probability-label">
                  <span>Fake News</span>
                  <strong>
                    {fakeProbability.toFixed(1)}%
                  </strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress fake-progress"
                    style={{
                      width: `${fakeProbability}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="probability-item">
                <div className="probability-label">
                  <span>Real News</span>
                  <strong>
                    {realProbability.toFixed(1)}%
                  </strong>
                </div>

                <div className="progress-track">
                  <div
                    className="progress real-progress"
                    style={{
                      width: `${realProbability}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="analysis-info">
              <span>ANALYSIS TYPE</span>

              <strong>
                {mode === "text"
                  ? "TEXT ANALYSIS"
                  : "IMAGE ANALYSIS"}
              </strong>
            </div>

            {result.ocr_text && mode === "image" && (
              <div className="ocr-box">
                <span>OCR DETECTED TEXT</span>
                <p>{result.ocr_text}</p>
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default App;