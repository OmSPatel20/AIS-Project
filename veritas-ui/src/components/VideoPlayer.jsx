// src/components/VideoPlayer.jsx
import React, { useRef, useEffect, useState } from "react";

/**
 * Props:
 *  - videoUrl
 *  - gradcams: array of { frame_index, base64 }
 */
export default function VideoPlayer({ videoUrl, gradcams = [], onTimeUpdate = () => {} }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [opacity, setOpacity] = useState(0.6);
  const [selectedCamIndex, setSelectedCamIndex] = useState(0);

  // make canvas match rendered video size
  function resizeCanvasToVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    canvas.width = Math.floor(rect.width);
    canvas.height = Math.floor(rect.height);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
  }

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onLoaded = () => setDuration(v.duration || 0);
    const onTime = () => {
      setCurrentTime(v.currentTime);
      onTimeUpdate(v.currentTime);
      renderOverlay();
    };
    const onResize = () => resizeCanvasToVideo();
    v.addEventListener("loadedmetadata", onLoaded);
    v.addEventListener("timeupdate", onTime);
    window.addEventListener("resize", onResize);
    // initial size
    setTimeout(resizeCanvasToVideo, 150);
    return () => {
      v.removeEventListener("loadedmetadata", onLoaded);
      v.removeEventListener("timeupdate", onTime);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line
  }, [videoRef.current, gradcams]);

  useEffect(() => {
    renderOverlay();
    // eslint-disable-next-line
  }, [selectedCamIndex, opacity]);

  function renderOverlay() {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const camObj = gradcams[selectedCamIndex] || gradcams[0];
    if (!camObj) return;
    const b64 = camObj.base64 || camObj.b64 || camObj;
    const src = b64?.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      ctx.globalAlpha = opacity;
      // draw image stretched to canvas so it aligns over video
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    };
    img.src = src;
  }

  return (
    <div className="video-card bg-black rounded relative">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="block"
        style={{ maxWidth: "100%", maxHeight: "100%" }}
      />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
      <div className="absolute left-3 bottom-3 right-3 flex items-center gap-3 bg-black bg-opacity-40 p-2 rounded z-20">
        <div className="text-white text-sm">Overlay opacity</div>
        <input
          className="w-44"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
        />
        <div className="ml-auto text-xs text-white opacity-90">Time: {Math.round(currentTime)}s / {Math.round(duration)}s</div>
      </div>

      {/* thumbs for gradcams */}
      {gradcams.length > 0 && (
        <div className="absolute right-3 top-3 z-30 flex gap-2">
          {gradcams.map((g, i) => {
            const b64 = g.base64 || g.b64 || g;
            const src = b64?.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
            return (
              <button key={i} onClick={() => setSelectedCamIndex(i)} className={`border rounded overflow-hidden ${selectedCamIndex===i ? "ring-2 ring-indigo-400" : ""}`}>
                <img src={src} alt={`cam-${i}`} className="w-20 h-14 object-cover" />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
