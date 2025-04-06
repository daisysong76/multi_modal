
import cv2
import os
import json
import whisper
from transformers import pipeline

def summarize_frame(frame):
    # Placeholder for actual visual summary
    return "Scene with visual elements."

def extract_audio(video_path, audio_path="temp_audio.wav"):
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}")
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def video_pipeline(video_path, output_path="video_output.json"):
    cap = cv2.VideoCapture(video_path)
    scene_data = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    count = 0

    while success:
        if count % int(frame_rate * 2) == 0:  # every 2 seconds
            summary = summarize_frame(frame)
            scene_data.append({"frame": count, "summary": summary})
        success, frame = cap.read()
        count += 1

    audio_path = extract_audio(video_path)
    audio_transcript = transcribe_audio(audio_path)

    output = {
        "scene_summaries": scene_data,
        "transcript": audio_transcript
    }
    with open(output_path, "w") as f:
        json.dump(output, f)

# Example usage
# video_pipeline("example.mp4")
