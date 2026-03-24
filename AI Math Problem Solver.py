"""
🧠 AI Math Solver — PyCharm Ready
Supports: Text | Image | Audio
"""

import os
from google import genai
from PIL import Image
import gradio as gr

# ─────────────────────────────
# OPTIONAL: Speech Recognition
# ─────────────────────────────
try:
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ─────────────────────────────
# CONFIG (UPDATED API KEY)
# ─────────────────────────────
API_KEY = "AIzaSyAu-9TXsWLKJ-ELnQT_E1SBLgi0JvrVI98"

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.0-flash"

# ─────────────────────────────
# PROMPT
# ─────────────────────────────
PROMPT = """
You are an expert math teacher.

Solve the problem step-by-step clearly.

Rules:
1. Start with "Given:"
2. Number each step
3. Show operations clearly
4. End with: ✅ Final Answer

Problem:
{input}
"""

# ─────────────────────────────
# AUDIO → TEXT
# ─────────────────────────────
def transcribe_audio(audio_path):
    if not AUDIO_AVAILABLE:
        return None, "❌ SpeechRecognition not installed"

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text, None
    except Exception as e:
        return None, f"❌ Audio error: {str(e)}"

# ─────────────────────────────
# GEMINI FUNCTIONS
# ─────────────────────────────
def solve_text(text):
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(input=text)
    )
    return response.text


def solve_image(image_path):
    img = Image.open(image_path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            "Read and solve this math problem step by step",
            img
        ]
    )
    return response.text


# ─────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────
def solve(input_text, audio_input, image_input):

    try:
        # Priority: Image > Audio > Text

        if image_input:
            result = solve_image(image_input)
            return f"📸 Image Input\n\n{result}"

        if audio_input:
            text, err = transcribe_audio(audio_input)
            if err:
                return err

            result = solve_text(text)
            return f"🎤 Audio Input\n📝 Transcribed: {text}\n\n{result}"

        if input_text and input_text.strip():
            result = solve_text(input_text)
            return f"✍️ Text Input\n\n{result}"

        return "⚠️ Please provide text, audio, or image input."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─────────────────────────────
# UI (GRADIO)
# ─────────────────────────────
with gr.Blocks() as app:

    gr.Markdown("# 🧠 AI Math Solver")

    text_input = gr.Textbox(
        label="✍️ Enter Equation",
        placeholder="e.g. 2x + 5 = 15"
    )

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="🎤 Record or Upload Audio"
    )

    image_input = gr.Image(
        sources=["upload", "webcam"],
        type="filepath",
        label="🖼️ Upload Image"
    )

    output = gr.Textbox(label="Solution", lines=20)

    solve_btn = gr.Button("🚀 Solve")

    solve_btn.click(
        solve,
        inputs=[text_input, audio_input, image_input],
        outputs=output
    )

app.launch()