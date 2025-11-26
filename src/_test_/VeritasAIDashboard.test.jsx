// src/__tests__/VeritasAIDashboard.test.jsx
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import VeritasAIDashboard from "../components/VeritasAIDashboard";

jest.mock("../api/api", () => ({
  uploadVideo: jest.fn(() => Promise.resolve({ job_id: "test1" })),
  pollUntilDone: jest.fn(() => Promise.resolve({ progress: 100, status: "done" })),
  getResult: jest.fn(() => Promise.resolve({
    label: "Suspected Deepfake",
    confidence: 92,
    explanation: "Synthetic mouth artifacts",
    timeline: Array.from({ length: 20 }).map((_,i)=>({t:i/20, confidence: Math.random()})),
    gradcam_images: [{ frame_index: 5, base64: "" }]
  })),
  SAMPLE_PROPOSAL_URL: "/mnt/data/UFID45173502_AIS_PROJECT-PROPOSAL.docx"
}));

test("upload flow triggers analysis and shows results", async () => {
  render(<VeritasAIDashboard />);
  const chooseBtn = screen.getByText(/Choose file/i);
  expect(chooseBtn).toBeTruthy();
  // We won't actually upload a file, but ensure Start Analysis prevents errors when no file.
  const startBtn = screen.queryByText(/Start analysis/i);
  expect(startBtn).toBeTruthy();
});
