import os
from dotenv import load_dotenv
from match_extractor import MatchProcessor

load_dotenv()

# ─── Match Keys ───────────────────────────────────────────────────────────────────
# Add match keys you want to process here.
MATCH_KEYS = [
    "2026casac_qm1",
    "2026casac_qm2",
]

# ─── Config ───────────────────────────────────────────────────────────────────────
CONFIG = {
    "tba_key":            os.getenv("TBA_KEY", ""),
    "mongo_uri":          os.getenv("MONGO_URI", ""),
    "comp_id":            "2026casac",
    "output_dir":         "output",
    "extract_scoreboard": True,
    "save_video_clip":    True,
    "upload_to_mongo":    False,
}

if __name__ == "__main__":
    comp_id    = CONFIG["comp_id"]
    output_dir = CONFIG["output_dir"]

    scoreboard_dir = os.path.join(output_dir, f"{comp_id}_scoreboards") if CONFIG["extract_scoreboard"] else None
    video_dir      = os.path.join(output_dir, f"{comp_id}_videos")      if CONFIG["save_video_clip"]    else None

    if scoreboard_dir:
        os.makedirs(scoreboard_dir, exist_ok=True)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    processor = MatchProcessor(
        tba_key=CONFIG["tba_key"],
        mongo_uri=CONFIG["mongo_key"] if CONFIG["upload_to_mongo"] else None,
    )

    try:
        for match_key in MATCH_KEYS:
            scoreboard_path = os.path.join(scoreboard_dir, f"{match_key}.json") if scoreboard_dir else None
            video_path      = os.path.join(video_dir,      f"{match_key}.mp4")  if video_dir      else None

            processor.process_match(
                match_key,
                save_video_path=video_path,
                save_scoreboard_path=scoreboard_path,
                upload_to_mongo=CONFIG["upload_to_mongo"],
            )
    finally:
        processor.shutdown()
        print("\nAll done.")
