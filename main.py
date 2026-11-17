import os
import subprocess
import uuid

import yt_dlp
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
CLIPS = os.path.join(BASE, "clips")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(CLIPS, exist_ok=True)

app = FastAPI()
app.mount("/clips", StaticFiles(directory=CLIPS), name="clips")

# "base" is fast. Use "small" for better accuracy (slower).
model = WhisperModel("base", device="cpu", compute_type="int8")


@app.get("/")
def home():
    return _page("index.html")


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def download_video(url, job):
    opts = {
        "outtmpl": os.path.join(UPLOADS, job + ".%(ext)s"),
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloads = info.get("requested_downloads")
        if downloads:
            return downloads[0]["filepath"]
        return ydl.prepare_filename(info)


def pick_moments(segments, count, length):
    windows, cur = [], None
    for seg in segments:
        text = seg.text.strip()
        if cur is None:
            cur = {"start": seg.start, "end": seg.end, "text": text}
        else:
            cur["end"] = seg.end
            cur["text"] += " " + text
        if cur["end"] - cur["start"] >= length:
            windows.append(cur)
            cur = None
    if cur and cur["end"] - cur["start"] >= length / 2:
        windows.append(cur)

    for w in windows:
        dur = max(w["end"] - w["start"], 1)
        words = len(w["text"].split())
        w["score"] = words / dur + 0.5 * w["text"].count("?") + 0.3 * w["text"].count("!")

    windows.sort(key=lambda w: w["score"], reverse=True)
    top = windows[:count]
    top.sort(key=lambda w: w["start"])
    return top


@app.post("/process")
def process(
    file: UploadFile = File(None),
    url: str = Form(""),
    count: int = Form(3),
    length: int = Form(45),
):
    job = uuid.uuid4().hex[:8]

    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        src = os.path.join(UPLOADS, job + ext)
        with open(src, "wb") as f:
            f.write(file.file.read())
    elif url.strip():
        try:
            src = download_video(url.strip(), job)
        except Exception as e:
            return {"error": "Could not download that link. Check it is a public video link. (" + str(e)[:150] + ")"}
    else:
        return {"error": "Choose a file or paste a link first."}

    segments, _ = model.transcribe(src)
    segments = list(segments)
    if not segments:
        return {"error": "No speech found in this video."}

    moments = pick_moments(segments, count, length)
    clips = []
    for i, m in enumerate(moments, 1):
        name = f"{job}_{i}.mp4"
        out = os.path.join(CLIPS, name)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(m["start"]), "-i", src,
            "-t", str(m["end"] - m["start"]),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", out,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"error": "FFmpeg failed. Check that ffmpeg -version works in cmd."}
        clips.append({
            "label": m["text"][:70],
            "range": f"{fmt(m['start'])} to {fmt(m['end'])}",
            "url": "/clips/" + name,
        })
    return {"clips": clips}
