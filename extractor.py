import cv2
import re
import numpy as np
import pytesseract
import json
import os
from pymongo import MongoClient
from vidgear.gears import CamGear
from concurrent.futures import ThreadPoolExecutor

SCOREBOARD_CONFIG = {
    "stream_url": "https://www.youtube.com/watch?v=d-hw7nU4pI0",
    "comp_id": "2026casac",
    "day": "1",
    "user": "doozer_server",
    "upload_to_mongo": True,
    "mongo_connection": "",
}

DEBUG = False
SAVE_VIDEOS = True
SHOW_VIDEO = False

comp_id = SCOREBOARD_CONFIG["comp_id"]
OUTPUT_DIR = f"{comp_id}_{SCOREBOARD_CONFIG['day']}_scoreboards"
VIDEO_DIR = f"{comp_id}_{SCOREBOARD_CONFIG['day']}_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Hardcoded ROIs at 1920x1080: (y1, y2, x1, x2)
BASE_ROIS = {
    "name":    (19,  55,  735,  1175),
    "red":     (65,  122, 1025, 1133),
    "timer":   (63,  124, 902,  1017),
    "blue":    (67,  124, 785,  896),
    "blue_rp": (71,  110, 83,   227),
    "red_rp":  (71,  109, 1738, 1881),
}

NUMBER_ROIS = {"red", "timer", "blue", "blue_rp", "red_rp"}

executor = ThreadPoolExecutor(max_workers=6)

UPLOAD_TO_MONGO = SCOREBOARD_CONFIG.get("upload_to_mongo", False)
mongo_collection = None
if UPLOAD_TO_MONGO:
    mongo_connection = SCOREBOARD_CONFIG.get("mongo_connection") or os.environ.get("MONGO_CONNECTION")
    if not mongo_connection:
        raise ValueError("upload_to_mongo is enabled but no mongo_connection set in config or MONGO_CONNECTION env var")
    _mongo_client = MongoClient(mongo_connection)
    mongo_collection = _mongo_client["doozervision"]["scoreboard_cache"]
    print("MongoDB connected.")


def upload_scoreboard(match_id, scoreboard_data):
    try:
        result = mongo_collection.update_one(
            {"match_id": match_id},
            {"$set": {"scoreboard_data": scoreboard_data["timeline"]}},
            upsert=True,
        )
        if result.upserted_id:
            print(f"MONGO: Inserted {match_id}")
        else:
            print(f"MONGO: Updated {match_id}")
    except Exception as e:
        print(f"MONGO ERROR: {e}")


def extract_text(roi, number_only=False):
    if roi is None or roi.size == 0:
        return ""
    config = "--psm 7"
    if number_only:
        config += " -c tessedit_char_whitelist=0123456789:"
    return pytesseract.image_to_string(roi, config=config).strip()


def parse_timer(timer_str):
    timer_str = timer_str.replace("O", "0").replace("l", "1").replace("S", "5").strip()
    try:
        if ":" in timer_str:
            parts = timer_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        return int(timer_str)
    except:
        return None


def has_changed(img1, img2, threshold=500):
    if img1 is None or img2 is None:
        return True
    return np.sum(cv2.absdiff(img1, img2)) > threshold


def parse_match_id(name):
    n = name.lower().strip()
    nums = re.findall(r'\d+', n)

    if "qualification" in n or "qual" in n:
        if nums:
            return f"{comp_id}_qm{nums[0]}"
    if "semifinal" in n or "semi" in n:
        if len(nums) >= 2:
            return f"{comp_id}_sf{nums[0]}m{nums[1]}"
        if nums:
            return f"{comp_id}_sf{nums[0]}m1"
    if "final" in n:
        if nums:
            return f"{comp_id}_f1m{nums[0]}"
    if "practice" in n:
        if nums:
            return f"{comp_id}_pm{nums[0]}"

    # Fallback: sanitize the raw name
    sanitized = re.sub(r'[^a-z0-9]+', '_', n).strip('_')
    return f"{comp_id}_{sanitized}"


stream = CamGear(source=SCOREBOARD_CONFIG["stream_url"], stream_mode=True, logging=True).start()

fps = 30
try:
    fps = stream.framerate
except:
    pass

cv2.imwrite(f"{comp_id}_savedframe.jpg", stream.read())
print(f"Using fps: {fps}")

in_match = False
current_timeline = []
current_match_name = ""
last_timer_sec = None
last_scores = (None, None)
missing_timer_count = 0
zero_timer_count = 0
video_writer = None

prev_rois = {k: None for k in BASE_ROIS}
last_ocr_results = {k: "" for k in BASE_ROIS}


while True:
    frame = stream.read()
    if frame is None:
        break

    curr_rois = {k: frame[y1:y2, x1:x2] for k, (y1, y2, x1, x2) in BASE_ROIS.items()}

    tasks = {}
    for key in curr_rois:
        if has_changed(curr_rois[key], prev_rois[key]):
            tasks[key] = executor.submit(extract_text, curr_rois[key], key in NUMBER_ROIS)
        else:
            tasks[key] = None

    for key, future in tasks.items():
        if future:
            last_ocr_results[key] = future.result()

    prev_rois = {k: v.copy() for k, v in curr_rois.items()}

    res_name   = last_ocr_results["name"]
    res_red    = last_ocr_results["red"]
    res_timer  = last_ocr_results["timer"]
    res_blue   = last_ocr_results["blue"]
    res_blue_rp = last_ocr_results["blue_rp"]
    res_red_rp  = last_ocr_results["red_rp"]
    sec = parse_timer(res_timer)

    if not in_match:
        if sec is not None and last_timer_sec is not None and len(res_name) > 10:
            if 0 < sec < last_timer_sec and (last_timer_sec - sec) < 10:
                in_match = True
                current_match_name = res_name
                current_timeline = []
                last_scores = (None, None)
                missing_timer_count = 0
                zero_timer_count = 0

                print(f"MATCH START: {res_name}")

                if SAVE_VIDEOS:
                    height, width = frame.shape[:2]
                    match_id = parse_match_id(res_name)
                    video_path = os.path.join(VIDEO_DIR, f"{match_id}.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
                    )
                    video_writer.write(frame)
    else:
        if SAVE_VIDEOS and video_writer is not None:
            video_writer.write(frame)

        if sec is not None:
            missing_timer_count = 0
            zero_timer_count = zero_timer_count + 1 if sec == 0 else 0

            if (res_red, res_blue) != last_scores:
                if res_red.isdigit() and res_blue.isdigit():
                    current_timeline.append({
                        "red": res_red,
                        "blue": res_blue,
                        "red_rp": res_red_rp,
                        "blue_rp": res_blue_rp,
                        "time": res_timer,
                    })
                    last_scores = (res_red, res_blue)
        else:
            missing_timer_count += 1

        if missing_timer_count > 40 or zero_timer_count > 40:
            in_match = False

            if video_writer is not None:
                video_writer.release()
                video_writer = None

            if len(current_timeline) > 10:
                match_id = parse_match_id(current_match_name)
                out_path = os.path.join(OUTPUT_DIR, f"{match_id}.json")
                with open(out_path, "w") as f:
                    json.dump({
                        "match_name": current_match_name,
                        "match_id": match_id,
                        "timeline": current_timeline,
                    }, f, indent=4)

                end_reason = "Missing Timer" if missing_timer_count > 40 else "Timer reached 0:00"
                print(f"SAVED: {match_id} | '{current_match_name}' (Ended via: {end_reason})")

                if UPLOAD_TO_MONGO:
                    upload_scoreboard(match_id, {"timeline": current_timeline})

    last_timer_sec = sec

    skip_amount = int(fps / 2) if not in_match else int(fps / 6)
    for _ in range(max(0, skip_amount - 1)):
        skipped_frame = stream.read()
        if skipped_frame is None:
            break
        if SAVE_VIDEOS and in_match and video_writer is not None:
            video_writer.write(skipped_frame)

    if SHOW_VIDEO:
        if DEBUG:
            for roi_name, roi_frame in curr_rois.items():
                cv2.imshow(roi_name, roi_frame)
        cv2.imshow("Doozer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if video_writer is not None:
    video_writer.release()

stream.stop()
cv2.destroyAllWindows()
executor.shutdown()
