import os
import cv2
import json
import math
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from pathlib import Path

import numpy as np

def sanitize_for_json(obj):
    """Convert numpy types (int64, float32, ndarray) to native Python types."""
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj


# ============ CONFIG ============
DATA_ROOT = Path(r"C:\Users\ospatel17\AIS Project\Data")            # folder containing original/ and deepfake/
OUTPUT_ROOT = Path(r"C:\Users\ospatel17\AIS Project\processed")     # where processed frames will be saved
FPS_EXTRACT = 1                     # extract 1 frame per second
IMG_SIZE = 224                      # final aligned face size
# =================================


# ========= FACE ALIGNMENT =============
def align_face(image, left_eye, right_eye, desired_size=224):
    desired_left_eye = (0.35, 0.35)

    # compute angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dY, dX))

    # determine scale
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_left_eye[0] - (1 - desired_left_eye[0])) * desired_size
    scale = desired_dist / dist

    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = desired_size * 0.5
    tY = desired_size * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned = cv2.warpAffine(image, M, (desired_size, desired_size), flags=cv2.INTER_CUBIC)
    return aligned
# =======================================


# ========== PROCESS SINGLE VIDEO ==========
def process_video(video_path, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(max(fps // FPS_EXTRACT, 1))

    detector = MTCNN()
    frame_idx = 0
    saved_frames = 0

    metadata = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb)

            if len(detections) > 0:
                det = detections[0]  # use highest confidence face
                box = det['box']
                keypoints = det['keypoints']

                x, y, w, h = box
                face = rgb[y:y+h, x:x+w]

                # alignment
                le = keypoints['left_eye']
                re = keypoints['right_eye']
                aligned = align_face(face, le, re, IMG_SIZE)
                aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

                out_path = output_folder / f"frame_{saved_frames:05d}.jpg"
                cv2.imwrite(str(out_path), aligned_bgr)

                metadata.append({
                    "frame": saved_frames,
                    "src_frame_idx": frame_idx,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "left_eye": le,
                    "right_eye": re
                })

                saved_frames += 1

        frame_idx += 1

    cap.release()

    # save metadata JSON
    meta_path = output_folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(metadata), f, indent=4)

    return saved_frames
# =========================================


# ========== MAIN PIPELINE ================
def run_preprocessing():
    categories = ["original", "Deepfakes"]

    for cat in categories:
        input_dir = DATA_ROOT / cat
        videos = list(input_dir.glob("*.mp4"))

        for vid in tqdm(videos, desc=f"Processing {cat}"):
            video_name = vid.stem
            out_dir = OUTPUT_ROOT / cat / video_name
            count = process_video(vid, out_dir)
            print(f"[OK] {video_name}: {count} aligned faces saved.")
# =========================================


if __name__ == "__main__":
    run_preprocessing()
