// src/components/UploadAnalyze.jsx
import React, { useRef, useState } from "react";
import { uploadVideo, SAMPLE_PROPOSAL_URL } from "../api";

export default function UploadAnalyze() {
  const fileRef = useRef(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function pickFile() {
    fileRef.current.click();
  }

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      alert("Select a video file (mp4, webm, mov).");
      return;
    }
    setFileInfo({ file: f, url: URL.createObjectURL(f) });
    setResult(null);
    setError(null);
    setUploadPct(0);
  }

  async function start() {
    if (!fileInfo?.file) return alert("Choose a video first.");
    setBusy(true);
    setResult(null);
    setError(null);
    setUploadPct(0);
    try {
      const res = await uploadVideo(fileInfo.file, (pct) => setUploadPct(Math.round(pct)));
      // backend expected to return JSON { prediction, confidence, gradcam_images, per_frame, ... }
      setResult(res);
    } catch (err) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  function exportReport() {
    if (!result) return alert("No result to export");
    const report = {
      videoName: fileInfo?.file?.name || "uploaded_video",
      analysis: result,
      provenance: SAMPLE_PROPOSAL_URL,
      generated_at: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.videoName}_veritas_report.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">VeritasAI — Deepfake Detector</h1>

      <div className="bg-white p-4 rounded shadow mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-600">Upload a video to analyze</div>
            <div className="mt-2 text-sm text-gray-700">{fileInfo?.file?.name || "No file selected"}</div>
          </div>
          <div className="flex gap-2">
            <input ref={fileRef} type="file" accept="video/*" onChange={onPick} className="hidden" />
            <button onClick={pickFile} className="px-3 py-2 bg-blue-600 text-white rounded">Choose file</button>
            <button onClick={() => { setFileInfo(null); setResult(null); setUploadPct(0); }} className="px-3 py-2 bg-gray-200 rounded">Clear</button>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button onClick={start} disabled={!fileInfo || busy} className={`px-4 py-2 rounded ${busy ? "bg-yellow-400" : "bg-green-600 text-white"}`}>
            {busy ? "Uploading..." : "Start Analysis"}
          </button>
          <div className="text-sm text-gray-500">Upload progress: {uploadPct}%</div>
        </div>
      </div>

      {error && <div className="bg-red-100 text-red-800 p-3 rounded mb-4">Error: {error}</div>}

      {result && (
        <div className="bg-white p-4 rounded shadow mb-6">
          <h2 className="text-lg font-semibold">Result</h2>
          <div className="mt-2 grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm">Label</div>
              <div className="text-xl font-bold">{result.prediction || result.label || "Unknown"}</div>
              <div className="text-sm text-gray-600 mt-1">Confidence: {result.confidence ?? result.confidence_percent ?? Math.round((result.confidence_float||0)*100)}%</div>
              <div className="mt-2 text-sm text-gray-700">{result.explanation || result.detail || ""}</div>

              <div className="mt-3 flex gap-2">
                <button onClick={exportReport} className="px-3 py-2 bg-indigo-600 text-white rounded">Export JSON</button>
              </div>
            </div>

            <div>
              <div className="text-sm">Frames processed</div>
              <div className="text-lg font-medium">{result.frames_processed ?? (result.per_frame && result.per_frame.length) ?? "N/A"}</div>

              <div className="mt-3">
                <div className="text-sm mb-2">Per-frame (first 10)</div>
                <div className="text-xs text-gray-600">
                  {(result.per_frame || []).slice(0,10).map((p, idx) => (
                    <div key={idx} className="flex justify-between border-b py-1">
                      <div>frame {p.src_frame_idx}</div>
                      <div>{Math.round((p.score ?? 0) * 100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Grad-CAM images */}
          <div className="mt-4">
            <div className="text-sm font-medium mb-2">Grad-CAM overlays</div>
            <div className="grid grid-cols-3 gap-3">
              {(result.gradcam_images || []).map((g, i) => {
                const b64 = g.base64 || g.b64 || g;
                const src = b64?.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
                return (
                  <div key={i} className="border rounded overflow-hidden">
                    <img alt={`gradcam-${i}`} src={src} className="w-full h-40 object-contain bg-black" />
                    <div className="p-2 text-xs">{typeof g.frame_index !== "undefined" ? `frame ${g.frame_index}` : ""}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div className="text-xs text-gray-500">
        Backend: {process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000"} — ensure Flask is running and CORS enabled.
      </div>
    </div>
  );
}
