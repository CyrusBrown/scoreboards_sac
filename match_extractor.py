import os
import json
import requests
import cv2
import pytesseract
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from pytubefix import YouTube

# ─── ROI Definitions (1920x1080) ─────────────────────────────────────────────────
_BASE_ROIS = {
    "name":    (19,  55,  735,  1175),
    "red":     (65,  122, 1025, 1133),
    "timer":   (63,  124, 902,  1017),
    "blue":    (67,  124, 785,  896),
    "blue_rp": (71,  110, 83,   227),
    "red_rp":  (71,  109, 1738, 1881),
}
_NUMBER_ROIS = {"red", "timer", "blue", "blue_rp", "red_rp"}
_TBA_BASE = "https://www.thebluealliance.com/api/v3"


class MatchProcessor:
    def __init__(self, tba_key, mongo_uri=None):
        self.tba_key = tba_key
        self.mongo_uri = mongo_uri
        self._executor = ThreadPoolExecutor(max_workers=6)

    # ─── TBA ──────────────────────────────────────────────────────────────────────

    def _tba_get(self, endpoint):
        url = f"{_TBA_BASE}/{endpoint.lstrip('/')}"
        try:
            r = requests.get(url, headers={"X-TBA-Auth-Key": self.tba_key})
            if r.status_code == 200:
                return r.json()
            print(f"TBA error {r.status_code} for {endpoint}")
        except Exception as e:
            print(f"TBA request failed: {e}")
        return None

    def get_youtube_id(self, match_key):
        data = self._tba_get(f"/match/{match_key}")
        if not data or "videos" not in data:
            return None
        for v in data["videos"]:
            if v["type"] == "youtube":
                return v["key"]
        return None

    # ─── Video ────────────────────────────────────────────────────────────────────

    def _get_video_cap(self, youtube_id):
        yt = YouTube(f"https://www.youtube.com/watch?v={youtube_id}")
        stream = yt.streams.filter(adaptive=True, file_extension="mp4").first()
        return cv2.VideoCapture(stream.url)

    # ─── OCR Helpers ──────────────────────────────────────────────────────────────

    def _extract_text(self, roi, number_only=False):
        if roi is None or roi.size == 0:
            return ""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7"
        if number_only:
            config += " -c tessedit_char_whitelist=0123456789:"
        return pytesseract.image_to_string(thresh, config=config).strip()

    def _has_changed(self, img1, img2, threshold=2.0):
        if img1 is None or img2 is None:
            return True
        return np.mean(cv2.absdiff(img1, img2)) > threshold

    def _parse_timer(self, timer_str):
        timer_str = timer_str.replace("O", "0").replace("l", "1").replace("S", "5").strip()
        try:
            if ":" in timer_str:
                parts = timer_str.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            return int(timer_str)
        except:
            return None

    def _scale_rois(self, frame):
        h, w = frame.shape[:2]
        sx, sy = w / 1920, h / 1080
        return {
            k: (int(y1 * sy), int(y2 * sy), int(x1 * sx), int(x2 * sx))
            for k, (y1, y2, x1, x2) in _BASE_ROIS.items()
        }

    # ─── MongoDB ──────────────────────────────────────────────────────────────────

    def _upload_to_mongo(self, match_id, timeline):
        if not self.mongo_uri:
            raise ValueError("No mongo_uri configured")
        client = MongoClient(self.mongo_uri)
        collection = client["doozervision"]["scoreboard_cache"]
        try:
            result = collection.update_one(
                {"match_id": match_id},
                {"$set": {"scoreboard_data": timeline}},
                upsert=True,
            )
            if result.upserted_id:
                print(f"MONGO: Inserted {match_id}")
            else:
                print(f"MONGO: Updated {match_id}")
        except Exception as e:
            print(f"MONGO ERROR: {e}")
        finally:
            client.close()

    # ─── Core ─────────────────────────────────────────────────────────────────────

    def process_match(self, match_key,
                      save_video_path=None,
                      save_scoreboard_path=None,
                      upload_to_mongo=False,
                      on_progress=None):
        """
        Process a single match by TBA key.

        Args:
            match_key:           TBA match key, e.g. "2026casac_qm1"
            save_video_path:     If set, write a video clip to this path
            save_scoreboard_path: If set, write scoreboard JSON to this path
            upload_to_mongo:     If True, upsert timeline to MongoDB
            on_progress:         Optional callback(current_frame, total_frames, message)

        Returns:
            list of timeline entries, or None on failure
        """
        def progress(frame, total, msg):
            print(msg)
            if on_progress:
                on_progress(frame, total, msg)

        progress(0, 0, f"Fetching TBA data for {match_key}...")
        yt_id = self.get_youtube_id(match_key)
        if not yt_id:
            progress(0, 0, f"No YouTube video found on TBA for {match_key}")
            return None

        progress(0, 0, f"Opening video (YouTube ID: {yt_id})...")
        cap = self._get_video_cap(yt_id)
        if not cap.isOpened():
            progress(0, 0, f"Failed to open video for {match_key}")
            return None

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip   = 15
        current_frame = 0
        timeline     = []
        last_scores  = (None, None)
        prev_rois    = {k: None for k in _BASE_ROIS}
        last_ocr     = {k: "" for k in _BASE_ROIS}
        rois         = None
        video_writer = None

        while True:
            ret, frame = cap.read()
            current_frame += 1
            if not ret:
                break

            if rois is None:
                h, w = frame.shape[:2]
                rois = self._scale_rois(frame)
                if save_video_path:
                    os.makedirs(os.path.dirname(save_video_path) or ".", exist_ok=True)
                    video_writer = cv2.VideoWriter(
                        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )

            if video_writer:
                video_writer.write(frame)

            curr_rois = {k: frame[y1:y2, x1:x2] for k, (y1, y2, x1, x2) in rois.items()}

            tasks = {}
            for key in curr_rois:
                if self._has_changed(curr_rois[key], prev_rois[key]):
                    tasks[key] = self._executor.submit(
                        self._extract_text, curr_rois[key], key in _NUMBER_ROIS
                    )
                else:
                    tasks[key] = None
            for key, future in tasks.items():
                if future:
                    last_ocr[key] = future.result()

            prev_rois = {k: v.copy() for k, v in curr_rois.items()}

            res_red     = last_ocr["red"]
            res_blue    = last_ocr["blue"]
            res_timer   = last_ocr["timer"]
            res_blue_rp = last_ocr["blue_rp"]
            res_red_rp  = last_ocr["red_rp"]
            sec = self._parse_timer(res_timer)

            if sec is not None and res_red.isdigit() and res_blue.isdigit():
                if (res_red, res_blue) != last_scores:
                    timeline.append({
                        "red":      res_red,
                        "blue":     res_blue,
                        "time":     res_timer,
                        "blue_rp":  res_blue_rp,
                        "red_rp":   res_red_rp,
                        "frame":    current_frame,
                        "timestamp": round(current_frame / fps, 2),
                    })
                    last_scores = (res_red, res_blue)
                    progress(
                        current_frame, total_frames,
                        f"Score update — Red: {res_red}  Blue: {res_blue}  ({res_timer})"
                    )

            # Skip frames
            for _ in range(frame_skip):
                if video_writer:
                    ret2, skip_frame = cap.read()
                    if not ret2:
                        break
                    video_writer.write(skip_frame)
                else:
                    if not cap.grab():
                        break
                current_frame += 1

            if current_frame % 300 == 0:
                pct = f" ({100 * current_frame // total_frames}%)" if total_frames else ""
                progress(
                    current_frame, total_frames,
                    f"Processing frame {current_frame}{f'/{total_frames}' if total_frames else ''}{pct}"
                )

        cap.release()
        if video_writer:
            video_writer.release()

        progress(
            total_frames or current_frame, total_frames or current_frame,
            f"Done: {match_key} — {len(timeline)} scoreboard entries"
        )

        if save_scoreboard_path and timeline:
            os.makedirs(os.path.dirname(save_scoreboard_path) or ".", exist_ok=True)
            with open(save_scoreboard_path, "w") as f:
                json.dump({"match_id": match_key, "timeline": timeline}, f, indent=4)
            print(f"Saved: {save_scoreboard_path}")

        if upload_to_mongo and timeline:
            self._upload_to_mongo(match_key, timeline)

        return timeline

    def shutdown(self):
        self._executor.shutdown(wait=False)
