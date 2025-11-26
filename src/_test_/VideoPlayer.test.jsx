// src/__tests__/VideoPlayer.test.jsx
import React from "react";
import { render, screen } from "@testing-library/react";
import VideoPlayer from "../components/VideoPlayer";

test("renders video player with canvas", () => {
  render(<VideoPlayer src={""} gradcamImages={[]} gradcamOpacity={0.5} />);
  const video = screen.getByRole("video", { hidden: true }) || document.querySelector("video");
  expect(video).toBeTruthy();
  const canvas = document.querySelector("canvas");
  expect(canvas).toBeTruthy();
});
