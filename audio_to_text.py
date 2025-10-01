from flask import Flask, request, jsonify
from flask_cors import CORS
from gtts import gTTS
from transformers import pipeline
import google.generativeai as genai
import os
from gtts.tts import gTTSError
from dotenv import load_dotenv
import psycopg2
from datetime import datetime, timezone, timedelta
import threading
import re

# Set the Indian Standard Time (IST) timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load environment variables from a .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)
conn.autocommit = True

# Load emotion analysis model from Hugging Face
emotion_analyzer = pipeline(
    task="sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)


# Helper function to save journal asynchronously
def save_to_db(user_message, scores):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mood_logs (
                    user_message, joy, sadness, anger, fear, surprise, disgust, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_message,
                    scores.get("joy", 0),
                    scores.get("sadness", 0),
                    scores.get("anger", 0),
                    scores.get("fear", 0),
                    scores.get("surprise", 0),
                    scores.get("disgust", 0),
                    datetime.now(IST),
                ),
            )
    except Exception as e:
        print("❌ Async DB save error:", e)


@app.route("/respond", methods=["POST"])
def respond():
    """
    Handles user messages, performs emotion analysis, generates a reply,
    converts the reply to speech, and returns it immediately.
    Journal saving happens asynchronously.
    """
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message.", "audio": None}), 400

    # Emotion analysis
    try:
        result = emotion_analyzer(user_input)
        scores = {item["label"].lower(): item["score"] for item in result[0]}
        summary_lines = [f"{label}: {score:.2f}" for label, score in scores.items()]
        score_summary = "\n".join(summary_lines)
    except Exception as e:
        print("❌ Emotion analysis error:", e)
        scores = {}
        score_summary = ""

    # Mood journaling summary
    try:
        prompt_gemini = f"Mood Journaling: Describe your feelings and emotions about '{user_input}' in 1–3 sentences. Focus on mood, stress, and emotions clearly and concisely."
        shorten_response = model.generate_content(prompt_gemini)
        shorten_input = shorten_response.text
    except Exception as e:
        print("❌ Gemini journaling error:", e)
        shorten_input = user_input  # fallback

    # Compassionate reply generation
    try:
        prompt_gemini = f"""
        You are a compassionate therapist chatbot.
        User message: "{user_input}"
        Detected emotions:
        {score_summary}

        Respond empathetically in 3-4 lines using emojis. Give supportive, scientifically-backed suggestions naturally, and gently ask a follow-up question to encourage the user to share more about how they feel. Keep the tone warm, friendly, and engaging.
        """

        response_gemini = model.generate_content(prompt_gemini)
        reply = response_gemini.text
    except Exception as e:
        print("❌ Gemini reply error:", e)
        reply = "I'm here for you, but something went wrong 😞. Please try again."

    # Text-to-Speech
    def remove_emojis(text):
        # Matches all non-text characters (emojis, symbols, etc.)
        return re.sub(r"[^\w\s.,!?'-]", "", text)

    clean_reply = remove_emojis(reply)
    audio_url = None
    try:
        tts = gTTS(text=clean_reply, lang="en", tld="co.uk")
        os.makedirs("static", exist_ok=True)
        audio_path = "static/response.mp3"
        tts.save(audio_path)
        audio_url = f"/{audio_path}"
    except gTTSError as e:
        print("🔴 gTTS failed:", e)

    # Save journal asynchronously
    threading.Thread(target=save_to_db, args=(shorten_input, scores)).start()

    # Return reply immediately
    return jsonify({"reply": reply, "audio": audio_url, "emotions": scores})


@app.route("/history", methods=["GET"])
def get_history():
    """Fetches the last 20 entries from the mood_logs table with rounded percentages and emojis."""
    try:
        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
        }

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_message, joy, sadness, anger, fear, surprise, disgust, created_at
                FROM mood_logs
                ORDER BY created_at DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()
            history = []
            for row in rows:
                emotions = {
                    emo: f"{round(row[i+1]*100)}% {emotion_emojis[emo]}"
                    for i, emo in enumerate(emotion_emojis.keys())
                }
                history.append(
                    {
                        "user": row[0],
                        "emotions": emotions,
                        "timestamp": row[7],
                    }
                )
        return jsonify(history)
    except Exception as e:
        print("❌ Fetch history error:", e)
        return jsonify({"error": "Unable to fetch history"}), 500


if __name__ == "__main__":
    app.run(debug=True)
