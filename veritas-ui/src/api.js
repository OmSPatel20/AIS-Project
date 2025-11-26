// src/api.js
export const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";
export const ENDPOINT_UPLOAD = API_BASE + "/analyze";
export const ENDPOINT_FEEDBACK = API_BASE + "/feedback";
export const ENDPOINT_EXPORT = API_BASE + "/export_report";

export async function uploadVideo(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", ENDPOINT_UPLOAD, true);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); } catch (err) { reject(new Error("Invalid JSON")); }
      } else { reject(new Error(`Server ${xhr.status}: ${xhr.statusText}\n${xhr.responseText}`)); }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress((e.loaded / e.total) * 100); };
    const form = new FormData(); form.append("video", file);
    xhr.send(form);
  });
}

export async function sendFeedback(payload) {
  const res = await fetch(ENDPOINT_FEEDBACK, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error("Failed to save feedback");
  return res.json();
}

export async function exportPdf(report) {
  const res = await fetch(ENDPOINT_EXPORT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report)
  });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}
