import os
import json
import threading
import queue
import time
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
from match_extractor import MatchProcessor

load_dotenv()

app = Flask(__name__)

TBA_KEY      = os.getenv("TBA_KEY", "")
MONGO_URI    = os.getenv("MONGO_URI", "")
UPLOAD_MONGO = os.getenv("UPLOAD_TO_MONGO", "false").lower() == "true"

# job_id -> { status, match_key, progress_queue, result, error }
_jobs = {}
_jobs_lock = threading.Lock()


def _run_job(job_id, match_key):
    q = _jobs[job_id]["progress_queue"]

    def on_progress(current_frame, total_frames, message):
        q.put({
            "type":    "progress",
            "current": current_frame,
            "total":   total_frames,
            "message": message,
        })

    try:
        processor = MatchProcessor(
            tba_key=TBA_KEY,
            mongo_uri=MONGO_URI if UPLOAD_MONGO else None,
        )
        timeline = processor.process_match(
            match_key,
            save_video_path=None,
            save_scoreboard_path=None,
            upload_to_mongo=UPLOAD_MONGO,
            on_progress=on_progress,
        )
        processor.shutdown()

        if timeline is None:
            msg = f"No YouTube video found on TBA for {match_key}, or failed to open stream."
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = msg
            q.put({"type": "error", "message": msg})
        else:
            _jobs[job_id]["result"] = {"match_id": match_key, "timeline": timeline}
            _jobs[job_id]["status"] = "done"
            q.put({"type": "done", "entry_count": len(timeline)})

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(e)
        q.put({"type": "error", "message": str(e)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    data      = request.get_json(force=True) or {}
    match_key = (data.get("match_key") or "").strip()
    if not match_key:
        return jsonify({"error": "match_key is required"}), 400

    job_id = f"{match_key}_{int(time.time() * 1000)}"
    with _jobs_lock:
        _jobs[job_id] = {
            "status":         "running",
            "match_key":      match_key,
            "progress_queue": queue.Queue(),
            "result":         None,
            "error":          None,
        }

    threading.Thread(target=_run_job, args=(job_id, match_key), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream_job(job_id):
    if job_id not in _jobs:
        return "Job not found", 404

    def generate():
        q = _jobs[job_id]["progress_queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<job_id>")
def download(job_id):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Not ready", 404

    match_key = job["match_key"]
    return Response(
        json.dumps(job["result"], indent=2),
        mimetype="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{match_key}_scoreboard.json"'
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
