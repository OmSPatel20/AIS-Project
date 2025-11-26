// src/components/Timeline.jsx
import React from "react";

/**
 * props:
 *  - data: array of numbers (0..1) confidence per frame
 *  - width, height: svg dims (optional)
 *  - onPick(index): callback when user clicks timeline
 *  - selected: currently selected index
 */
export default function Timeline({ data = [], width = 480, height = 64, onPick = () => {}, selected = 0 }) {
  if (!data || data.length === 0) {
    return <div className="text-sm text-gray-400">No timeline available</div>;
  }

  const w = width;
  const h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const norm = (v) => (max - min < 1e-6 ? 0.5 : (v - min) / (max - min));

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1 || 1)) * (w - 2) + 1;
    const y = h - (norm(v) * (h - 6) + 3);
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg
      width={w}
      height={h}
      className="cursor-pointer w-full"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const idx = Math.floor((x / w) * data.length);
        onPick(Math.max(0, Math.min(data.length - 1, idx)));
      }}
      role="img"
      aria-label="confidence timeline"
    >
      <defs>
        <linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#fb7185" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#f97316" stopOpacity="0.6" />
        </linearGradient>
      </defs>

      <polyline
        points={points}
        fill="none"
        stroke="#6b7280"
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* vertical indicator for selected */}
      {(() => {
        const selX = ((selected / (data.length - 1 || 1)) * (w - 2)) + 1;
        return <line x1={selX} x2={selX} y1={0} y2={h} stroke="#2563eb" strokeWidth={1} strokeDasharray="3 3" />;
      })()}

      {/* optional bars */}
      {data.map((v,i) => {
        const x = (i / (data.length - 1 || 1)) * (w - 2) + 1;
        const barH = Math.max(1, (norm(v) * (h - 6)));
        return <rect key={i} x={x-1} y={h-3-barH} width={2} height={barH} fill="#ef4444" opacity={0.45} />;
      })}
    </svg>
  );
}
