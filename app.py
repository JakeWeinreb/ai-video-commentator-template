import gradio as gr
import openai
import cv2
import tempfile
import os
import base64

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_key_frames(video_path, interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frames = []
    count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            _, buf = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buf).decode('utf-8'))
        count += 1
    cap.release()
    return frames[:5]  # Limit for quick processing

def generate_commentary(frames_b64):
    commentary = ""
    for i, img_b64 in enumerate(frames_b64):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sports commentator providing live play-by-play and color commentary."},
                {"role": "user", "content": f"Here's frame {i+1} of a video. Describe what you see and generate exciting commentary."},
                {"role": "user", "content": {"image": img_b64, "mime_type": "image/jpeg"}}
            ],
            max_tokens=300
        )
        commentary += f"Frame {i+1} Commentary:\n{response.choices[0].message.content}\n\n"
    return commentary

def process_video(video):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video.read())
        temp_path = temp.name

    frames = extract_key_frames(temp_path)
    result = generate_commentary(frames)
    return result

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a short video clip (5–10 seconds)", format="mp4"),
    outputs=gr.Textbox(label="AI Commentary"),
    title="AI Video Commentator",
    description="Upload a short video and get real-time AI-generated sports-style commentary."
)

iface.launch()
