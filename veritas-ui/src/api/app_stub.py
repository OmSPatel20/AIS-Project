# app_stub.py
from flask import Flask, request, jsonify
import uuid, threading, time, base64, os

app = Flask(__name__)
JOBS = {}

@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "video missing"}), 400
    job_id = str(uuid.uuid4())
    path = os.path.join("uploads", f.filename)
    os.makedirs("uploads", exist_ok=True)
    f.save(path)
    JOBS[job_id] = {"progress": 0, "status": "processing", "path": path}
    threading.Thread(target=simulate_process, args=(job_id, path)).start()
    return jsonify({"job_id": job_id})

@app.route("/api/status")
def status():
    job_id = request.args.get("job_id")
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify({"job_id": job_id, "progress": job.get("progress",0), "status": job.get("status","processing")})

@app.route("/api/result")
def result():
    job_id = request.args.get("job_id")
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error":"not ready"}), 404
    return jsonify(job["result"])

def simulate_process(job_id, path):
    # simulate work
    for p in range(0, 101, 10):
        JOBS[job_id]["progress"] = p
        time.sleep(0.5)
    # produce fake result and embed one dummy gradcam image as base64
    img_path = "static/dummy_gradcam.png"
    b64 = ""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    JOBS[job_id]["result"] = {
        "job_id": job_id,
        "label": "Suspected Deepfake",
        "confidence": 86,
        "explanation": "Simulated.",
        "timeline": [{"t": i/20, "confidence": 0.4+0.5*(i%5)/5} for i in range(20)],
        "gradcam_images": [{"frame_index": 5, "base64": b64}]
    }
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["progress"] = 100
