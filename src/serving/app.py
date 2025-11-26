# src/serving/app.py
import os
import io
import time
import json
import math
import base64
import threading
from pathlib import Path
from PIL import Image

from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from prometheus_client import Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
# Note: you already import generate_latest above -> keep consistent

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import glob
from sklearn.metrics import roc_auc_score
import random


# -------- CONFIG ----------
MODEL_PATH = os.environ.get("MODEL_PATH", str(Path.cwd() / "models" / "best_model.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_DIR = Path.cwd() / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR = Path.cwd() / "feedback"
REPORTS_DIR = Path.cwd() / "reports"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 224
FRAME_SAMPLE_FPS = 1
FRAME_LIMIT = 200
GRADCAM_MAX = 6
SAMPLE_PROPOSAL_URL = r"/mnt/data/UFID45173502_AIS_PROJECT-PROPOSAL.docx"  # your uploaded file path
# --------------------------

app = Flask(__name__)
CORS(app)
REQUESTS = Counter('veritasai_requests_total', 'Total requests')
INFER_LATENCY = Histogram('veritasai_request_latency_seconds', 'Latency seconds')
# Prometheus metrics we will export
AUC_GAUGE = Gauge("veritasai_video_auc", "Rolling AUC computed from feedback (0..1)")
AUC_LOWER_GAUGE = Gauge("veritasai_video_auc_lower", "Lower bound of bootstrap CI for AUC")
AUC_UPPER_GAUGE = Gauge("veritasai_video_auc_upper", "Upper bound of bootstrap CI for AUC")

# Histogram of per-video confidence scores (for quantiles)
# Note: use reasonably sized buckets (0.0..1.0)
CONF_HIST = Histogram("veritasai_confidence", "Per-video confidence histogram", buckets=[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0])


# ---- helpers ----
def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_frames(video_path, sample_rate_fps=1, max_frames=FRAME_LIMIT):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / max(1, sample_rate_fps))), 1)
    frames = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, rgb))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def align_face_rgb(face_rgb, left_eye, right_eye, desired_size=IMG_SIZE):
    left_eye = np.array(left_eye, dtype=np.float32)
    right_eye = np.array(right_eye, dtype=np.float32)
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_left_eye = (0.35, 0.35)
    if dist > 1e-6:
        scale = (desired_size * (1 - 2 * desired_left_eye[0])) / dist
    else:
        scale = 1.0
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = desired_size * 0.5
    tY = desired_size * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    aligned = cv2.warpAffine(face_rgb, M, (desired_size, desired_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned

# ---- model loading (adapt if your model arch differs) ----
def build_model_like_training():
    # default: EfficientNet-B0 with a small head (matches training script)
    backbone = torch.hub.load('pytorch/vision:v0.14.0', 'efficientnet_b0', pretrained=False)
    lin = next((m for m in backbone.classifier if isinstance(m, torch.nn.Linear)), None)
    in_features = lin.in_features if lin is not None else 1280
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(512, 1)
    )
    return backbone

def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        app.logger.warning(f"Model not found at {path}. inference will error until you provide model.")
        return None
    ckpt = torch.load(path, map_location="cpu")
    model = build_model_like_training()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

# init once
MODEL = load_model()
DETECTOR = MTCNN(keep_all=True, device=str(DEVICE))

# ---- gradcam helper ----
def generate_gradcams(model, saved_tensors_cpu, saved_rgb_norm, max_images=GRADCAM_MAX):
    cams = []
    if model is None:
        return cams
    try:
        # pick target conv layer heuristically
        target_layer = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
        if target_layer is None:
            target_layer = list(model.modules())[-1]
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(DEVICE.type=="cuda"))

        n = len(saved_tensors_cpu)
        if n == 0:
            return cams
        picks = list(range(0, n, max(1, n // max_images)))[:max_images]
        for idx in picks:
            try:
                inp = saved_tensors_cpu[idx].to(DEVICE)
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=inp, targets=targets)[0]
                rgb = saved_rgb_norm[idx]
                overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
                pil = Image.fromarray(overlay)
                cams.append({"frame_index": int(idx), "base64": pil_to_b64(pil)})
            except Exception as e:
                app.logger.warning(f"gradcam failed for idx {idx}: {e}")
                continue
    except Exception as e:
        app.logger.warning(f"gradcam generation failed: {e}")
    return cams

# ---- API routes ----
@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()
    REQUESTS.inc()

    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400
    f = request.files["video"]
    fname = f.filename or f"upload_{int(time.time())}.bin"
    safe_name = fname.replace(" ", "_")
    out_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    f.save(str(out_path))

    # extract frames
    frames = extract_frames(out_path, sample_rate_fps=FRAME_SAMPLE_FPS, max_frames=FRAME_LIMIT)
    if not frames:
        INFER_LATENCY.observe(time.time() - start)
        return jsonify({"error": "no frames extracted"}), 400

    # run detection + inference per frame
    per_scores = []
    per_meta = []
    saved_tensors = []
    saved_rgb = []

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    for (src_idx, rgb) in frames:
        # --- FACENET-PYTORCH MTCNN ---
        try:
            boxes, probs, landmarks = DETECTOR.detect(rgb, landmarks=True)
        except Exception as e:
            app.logger.warning(f"MTCNN.detect failed on frame {src_idx}: {e}")
            continue

        if boxes is None or len(boxes) == 0:
            continue

        # pick highest-confidence face
        if probs is None:
            sel = 0
        else:
            sel = int(np.argmax(probs))

        box = boxes[sel]  # [x1,y1,x2,y2]
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w, _ = rgb.shape

        # clip
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        face = rgb[y1:y2, x1:x2]

        # landmarks: (N,5,2) order = [left_eye, right_eye, nose, mouth_l, mouth_r]
        le = None
        re = None
        if landmarks is not None:
            try:
                lm = landmarks[sel]
                le = (float(lm[0][0]) - x1, float(lm[0][1]) - y1)
                re = (float(lm[1][0]) - x1, float(lm[1][1]) - y1)
            except Exception:
                le, re = None, None

        # fallback if eyes missing
        if le is None or re is None:
            fh, fw, _ = face.shape
            le = (fw * 0.3, fh * 0.35)
            re = (fw * 0.7, fh * 0.35)

        try:
            aligned = align_face_rgb(face, le, re, desired_size=IMG_SIZE)
        except Exception:
            aligned = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        tensor = transform(aligned).unsqueeze(0)
        saved_tensors.append(tensor.cpu())
        saved_rgb.append(aligned.astype(np.float32) / 255.0)

        if MODEL is None:
            prob = 0.5
        else:
            with torch.no_grad():
                out = MODEL(tensor.to(DEVICE))
                prob = float(torch.sigmoid(out).cpu().numpy().reshape(-1)[0])

        per_scores.append(prob)
        per_meta.append({
            "src_frame_idx": int(src_idx),
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "score": float(prob)
        })


    if not per_scores:
        INFER_LATENCY.observe(time.time() - start)
        return jsonify({"error": "no faces detected"}), 400

    # aggregate
    scores = np.array(per_scores)
    video_conf = float(scores.mean())
    pred_label = "Suspected Deepfake" if video_conf >= 0.5 else "Likely Authentic"

    # gradcams (limited)
    gradcams = generate_gradcams(MODEL, saved_tensors, saved_rgb, max_images=GRADCAM_MAX)

    result = {
        "video": out_path.name,
        "prediction": pred_label,
        "confidence": int(round(video_conf * 100)),
        "confidence_float": video_conf,
        "frames_processed": len(per_scores),
        "per_frame": per_meta,
        "gradcam_images": gradcams,
        "proposal_doc": SAMPLE_PROPOSAL_URL
    }

    INFER_LATENCY.observe(time.time() - start)
    return jsonify(result)
# after computing video_conf (0..1)
try:
    CONF_HIST.observe(video_conf)
except Exception as e:
    app.logger.debug("Failed to observe confidence histogram: %s", e)

# ----------------------------
# Feedback & PDF export APIs
# ----------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Save JSON feedback sent by the frontend to feedback/<video>_<ts>.json
    Expected JSON: { videoName, label, confidence, comments, timestamp }
    """
    try:
        data = request.get_json(force=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        safe_name = (data.get("videoName") or "unknown").replace(" ", "_")
        fname = FEEDBACK_DIR / f"{safe_name}_{ts}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return jsonify({"ok": True, "path": str(fname)}), 200
    except Exception as e:
        app.logger.exception("Feedback save failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/export_report", methods=["POST"])
def export_report():
    """
    Build a simple forensic PDF from the JSON payload and return it as an attachment.
    Expected JSON payload:
      {
        videoName: "...",
        analysis: { prediction, confidence, confidence_float, per_frame, gradcam_images, explanation, ... },
        provenance: "...",            # optional
        generated_at: "..."          # optional
      }
    """
    try:
        payload = request.get_json(force=True)
        video_name = payload.get("videoName", "uploaded_video")
        analysis = payload.get("analysis", {})
        prov = payload.get("provenance", SAMPLE_PROPOSAL_URL if 'SAMPLE_PROPOSAL_URL' in globals() else "")
        generated_at = payload.get("generated_at", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # create PDF in-memory
        bio = io.BytesIO()
        c = canvas.Canvas(bio, pagesize=letter)
        w, h = letter

        # Title & metadata
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, h - 60, "VeritasAI Forensic Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, h - 80, f"Video: {video_name}")
        c.drawString(40, h - 95, f"Generated: {generated_at}")
        if prov:
            c.drawString(40, h - 110, f"Provenance: {prov}")

        # Summary (label + confidence + explanation)
        label = analysis.get("prediction") or analysis.get("label", "Unknown")
        confidence = analysis.get("confidence")
        if confidence is None:
            # try confidence_float fallback
            confidence = int(round((analysis.get("confidence_float", 0)) * 100))
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h - 140, f"Label: {label}  ({confidence}%)")
        c.setFont("Helvetica", 11)
        explanation = analysis.get("explanation") or analysis.get("detail") or "No explanation provided by model."
        text = c.beginText(40, h - 165)
        text.setFont("Helvetica", 10)
        # wrap lines conservatively
        for paragraph in explanation.split("\n"):
            # split long lines into ~90-char chunks
            while len(paragraph) > 95:
                text.textLine(paragraph[:95])
                paragraph = paragraph[95:]
            text.textLine(paragraph)
        c.drawText(text)

        # Per-frame sample
        per_frame = analysis.get("per_frame") or []
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, h - 300, "Per-frame sample (first 10):")
        c.setFont("Helvetica", 10)
        y = h - 320
        for pf in per_frame[:10]:
            s = f"frame {pf.get('src_frame_idx')} score {int(round((pf.get('score', 0)) * 100))}%"
            c.drawString(45, y, s)
            y -= 14

        # Attach first gradcam image (if any)
        gradcams = analysis.get("gradcam_images") or []
        if len(gradcams) > 0:
            try:
                g0 = gradcams[0]
                # g0 may be dict with 'base64' or raw base64 string
                b64 = g0.get("base64") if isinstance(g0, dict) else g0
                if b64 and not b64.startswith("data:"):
                    b64 = "data:image/png;base64," + b64
                if b64:
                    header, b64data = b64.split(",", 1)
                    imgdata = base64.b64decode(b64data)
                    img = ImageReader(io.BytesIO(imgdata))
                    # draw at bottom-right
                    c.drawImage(img, w - 260, 60, width=200, height=140, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                app.logger.warning("Could not attach gradcam to PDF: %s", e)

        c.showPage()
        c.save()
        bio.seek(0)

        # persist the PDF on disk (use timestamped name)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        safe_video = video_name.replace(" ", "_")
        out_path = REPORTS_DIR / f"{safe_video}_{ts}.pdf"
        with open(out_path, "wb") as f:
            f.write(bio.getvalue())

        # return saved file as attachment
        return send_file(str(out_path), as_attachment=True, download_name=out_path.name, mimetype="application/pdf")

    except Exception as e:
        app.logger.exception("export_report failed")
        return jsonify({"ok": False, "error": str(e)}), 500
@app.route("/reports", methods=["GET"])
def list_reports():
    files = []
    for p in sorted(REPORTS_DIR.glob("*.pdf"), reverse=True):
        files.append({"name": p.name, "path": str(p), "url": f"/reports/download/{p.name}"})
    return jsonify(files)

@app.route("/reports/download/<fn>", methods=["GET"])
def download_report(fn):
    fpath = REPORTS_DIR / fn
    if not fpath.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(fpath), as_attachment=True, download_name=fpath.name)
def compute_auc_and_bootstrap(feedback_dir=FEEDBACK_DIR, n_bootstrap=1000, alpha=0.05):
    """
    Reads all feedback JSON files in feedback_dir, expects each file to have:
      { "videoName": "...", "label": 0 or 1 OR "label": "Suspected Deepfake"/"Likely Authentic",
        "confidence": integer percent OR "confidence_float": float }
    Computes AUC and bootstrap CI, updates Prometheus gauges.
    """
    feedback_files = sorted(feedback_dir.glob("*.json"))
    if not feedback_files:
        # nothing to compute
        return None

    y_true = []
    y_score = []

    for p in feedback_files:
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        # label may be numeric or textual
        lbl = data.get("label")
        if lbl is None:
            # legacy keys
            lbl = data.get("true_label") or data.get("gt") or data.get("ground_truth")
        # normalize label to 0/1 if textual
        if isinstance(lbl, str):
            lbl_norm = 1 if "fake" in lbl.lower() else 0
        else:
            lbl_norm = int(lbl) if lbl is not None else None

        if lbl_norm is None:
            continue

        # score: either confidence (0-100) or confidence_float (0..1)
        if data.get("confidence_float") is not None:
            score = float(data["confidence_float"])
        else:
            # integer percent -> convert to 0..1
            score = float(data.get("confidence", 0)) / 100.0

        y_true.append(lbl_norm)
        y_score.append(score)

    if len(y_true) < 2 or len(np.unique(y_true)) == 1:
        # Not enough data or no class variance
        return None

    # compute AUC
    try:
        auc_val = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc_val = None

    # bootstrap CI
    if auc_val is not None and len(y_true) >= 30:
        n = len(y_true)
        boot_vals = []
        rng = random.Random(12345)
        for _ in range(n_bootstrap):
            idx = [rng.randrange(0, n) for _ in range(n)]
            try:
                val = roc_auc_score([y_true[i] for i in idx], [y_score[i] for i in idx])
                boot_vals.append(val)
            except Exception:
                continue
        if boot_vals:
            lo = float(np.percentile(boot_vals, 100 * (alpha/2)))
            hi = float(np.percentile(boot_vals, 100 * (1 - alpha/2)))
        else:
            lo, hi = auc_val, auc_val
    else:
        # small-sample fallback: no CI or symmetric +/- 0.0
        lo, hi = auc_val, auc_val

    # set prometheus gauges (AUC in range 0..1)
    if auc_val is not None:
        AUC_GAUGE.set(auc_val)
        if lo is not None and hi is not None:
            AUC_LOWER_GAUGE.set(lo)
            AUC_UPPER_GAUGE.set(hi)

    return {"auc": auc_val, "lower": lo, "upper": hi, "n": len(y_true)}
# after writing JSON file in feedback endpoint:
res = compute_auc_and_bootstrap()
app.logger.info(f"AUC recomputed from feedback: {res}")
import threading, time

def periodic_metrics_update(interval_sec=300):
    while True:
        try:
            compute_auc_and_bootstrap()
        except Exception as e:
            app.logger.warning("periodic_metrics_update failed: %s", e)
        time.sleep(interval_sec)

# start thread once (after app creation)
t = threading.Thread(target=periodic_metrics_update, args=(300,), daemon=True)
t.start()


if __name__ == "__main__":
    # quick dev server
    app.run(host="0.0.0.0", port=8000, debug=True)
