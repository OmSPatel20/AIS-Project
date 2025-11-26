import React from "react";
import ReactDOM from "react-dom/client";
import VeritasAIDashboard from "./components/VeritasAIDashboard";
import "./index.css";

const container = document.getElementById("root");
const root = ReactDOM.createRoot(container);

root.render(
  <React.StrictMode>
    <VeritasAIDashboard />
  </React.StrictMode>
);
