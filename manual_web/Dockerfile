FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is sac_scoreboards/ so we can pull in the shared core module
COPY manual_web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY match_extractor.py .
COPY manual_web/ .

CMD ["python", "app.py"]
