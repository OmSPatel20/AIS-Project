// src/App.js
import React from "react";
import UploadAnalyze from "./components/UploadAnalyze";
import VeritasAIDashboard from "./components/VeritasAIDashboard";
import "./index.css"; // tailwind directives or your CSS

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <UploadAnalyze />
    </div>
  );
}
root.render(<VeritasAIDashboard />);

export default App;
