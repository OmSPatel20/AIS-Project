// src/components/VeritasAIDashboard.jsx
import React, { useRef, useState } from "react";
import { uploadVideo, sendFeedback, exportPdf } from "../api";
import VideoPlayer from "./VideoPlayer";
import Timeline from "./Timeline";

export default function VeritasAIDashboard() {
  const fileRef = useRef(null);
  const [fileObj, setFileObj] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [result, setResult] = useState(null);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackText, setFeedbackText] = useState("");
  const [feedbackSaved, setFeedbackSaved] = useState(false);
  const [error, setError] = useState(null);

  function pickFile() { fileRef.current.click(); }

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("video/")) { alert("Please select a video file."); return; }
    setFileObj({ file: f, url: URL.createObjectURL(f) });
    setResult(null); setUploadPct(0); setError(null); setFeedbackSaved(false);
  }

  async function startAnalysis() {
    if (!fileObj) return alert("Choose a video first");
    setBusy(true); setError(null); setResult(null); setUploadPct(0);
    try {
      const res = await uploadVideo(fileObj.file, (pct) => setUploadPct(Math.round(pct)));
      setResult(res);
      if (res && res.timeline) setSelectedFrameIndex(Math.floor(res.timeline.length / 2));
    } catch (err) { setError(err.message || String(err)); }
    finally { setBusy(false); }
  }

  async function submitFeedback() {
    if (!fileObj || !result) return alert("Provide feedback after analysis");
    const payload = {
      videoName: fileObj.file.name,
      label: result.prediction || result.label,
      confidence: result.confidence ?? Math.round((result.confidence_float || 0)*100),
      comments: feedbackText,
      timestamp: new Date().toISOString()
    };
    try {
      await sendFeedback(payload);
      setFeedbackSaved(true);
      setFeedbackText("");
      setFeedbackOpen(false);
      alert("Feedback saved.");
    } catch (err) { alert("Failed to save feedback: " + err.message); }
  }

  async function onExportPdf() {
    if (!result) return alert("No analysis to export");
    const report = {
      videoName: fileObj?.file?.name || "uploaded_video",
      analysis: result,
      generated_at: new Date().toISOString()
    };
    try {
      const blob = await exportPdf(report);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${report.videoName}_veritas_report.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) { alert("Export failed: " + err.message); }
  }

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-6xl mx-auto grid grid-cols-12 gap-6">
        {/* Left */}
        <div className="col-span-8">
          <div className="panel p-4" style={{ background: "transparent", boxShadow: "none" }}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-2xl font-semibold">Upload & analyze</h2>
                <div className="text-sm text-gray-200">Drop or pick a video then click Start Analysis</div>
              </div>
              <div className="flex gap-2 items-center">
                <input ref={fileRef} type="file" accept="video/*" onChange={onPick} className="hidden" />
                <button onClick={pickFile} className="btn btn-primary">Choose</button>
                <button onClick={() => { setFileObj(null); setResult(null); }} className="btn btn-ghost">Clear</button>
              </div>
            </div>

            <div className="mb-3 flex items-center gap-4">
              <button onClick={startAnalysis} disabled={!fileObj || busy} className={`btn ${busy ? "btn-neutral" : "btn-primary"}`}>
                {busy ? "Processing..." : "Start Analysis"}
              </button>
              <div className="text-sm text-gray-200">Upload: {uploadPct}%</div>
              {result && <button className="btn btn-primary ml-4" onClick={onExportPdf}>Export PDF</button>}
              <button className="btn btn-ghost ml-2" onClick={() => setFeedbackOpen(!feedbackOpen)}>{feedbackOpen ? "Close feedback" : "Feedback"}</button>
            </div>

            <div className="video-card mb-3">
              <VideoPlayer videoUrl={fileObj?.url} gradcams={(result && result.gradcam_images) || []} />
            </div>

            <div className="timeline-card p-3">
              <h4 className="text-sm font-medium text-white mb-2">Confidence timeline</h4>
              <Timeline
                data={(result && result.timeline && result.timeline.map(x => x.confidence)) || (result && result.per_frame && result.per_frame.map(x => x.score)) || []}
                width={600}
                height={72}
                onPick={(idx) => setSelectedFrameIndex(idx)}
                selected={selectedFrameIndex}
              />
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="col-span-4">
          <div className="panel p-4">
            <h3 className="text-xl font-semibold">Analysis Summary</h3>
            {error && <div className="text-sm text-red-400 my-2">{error}</div>}
            {result ? (
              <div className="mt-3 text-sm space-y-2">
                <div>Label: <strong>{result.prediction || result.label}</strong></div>
                <div>Top confidence: <strong>{result.confidence ?? Math.round((result.confidence_float||0)*100)}%</strong></div>
                <div className="text-gray-300">{result.explanation || ""}</div>
                <div className="text-xs text-gray-400 mt-2">Frames: {result.frames_processed ?? (result.per_frame && result.per_frame.length) ?? "N/A"}</div>
              </div>
            ) : (
              <div className="text-sm text-gray-300 mt-2">No analysis yet. Upload a video and start analysis.</div>
            )}
          </div>

          {feedbackOpen && (
            <div className="panel p-4 mt-4 feedback-box">
              <h4 className="text-md font-semibold">Feedback</h4>
              <textarea value={feedbackText} onChange={(e) => setFeedbackText(e.target.value)} placeholder="Why do you agree or disagree with the label?" />
              <div className="flex gap-2 mt-2">
                <button onClick={submitFeedback} className="btn btn-primary">Send feedback</button>
                <button onClick={() => { setFeedbackText(""); setFeedbackOpen(false); }} className="btn btn-ghost">Cancel</button>
              </div>
              {feedbackSaved && <div className="text-sm text-green-400 mt-2">Feedback saved.</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
