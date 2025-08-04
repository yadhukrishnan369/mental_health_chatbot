from flask import Flask, request, jsonify
from flask_cors import CORS
from gtts import gTTS
from transformers import pipeline
import google.generativeai as genai
import os
from gtts.tts import gTTSError
from dotenv import load_dotenv
import psycopg2
from datetime import datetime

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load env vars
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
conn.autocommit = True

# Load emotion model
emotion_analyzer = pipeline(
    task="sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

@app.route("/respond", methods=["POST"])
def respond():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message.", "audio": None}), 400

    # Emotion analysis
    try:
        result = emotion_analyzer(user_input)
        scores = {item['label'].lower(): item['score'] for item in result[0]}
    except Exception as e:
        print("❌ Emotion analysis error:", e)
        return jsonify({"reply": "Emotion analysis failed.", "audio": None}), 500

    # Gemini response
    try:
        score_summary = "\n".join([f"{item['label']}: {item['score']:.2f}" for item in result[0]])
        prompt = f"""You are a compassionate therapist chatbot.
User message: "{user_input}"
Detected emotions:
{score_summary}

Respond empathetically using emojis and give supportive suggestions in 2-3 lines."""
        response = model.generate_content(prompt)
        reply = response.text
    except Exception as e:
        print("❌ Gemini API error:", e)
        reply = "I'm here for you, but something went wrong 😞. Please try again."

    # Text to Speech
    audio_url = None
    try:
        tts = gTTS(text=reply, lang='en', tld='co.uk')
        os.makedirs("static", exist_ok=True)
        audio_path = "static/response.mp3"
        tts.save(audio_path)
        audio_url = f"/{audio_path}"
    except gTTSError as e:
        print("🔴 gTTS failed:", e)

    # Store in DB
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO mood_logs (
                    user_message, joy, sadness, anger, fear, surprise, disgust, bot_reply, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_input,
                scores.get('joy', 0),
                scores.get('sadness', 0),
                scores.get('anger', 0),
                scores.get('fear', 0),
                scores.get('surprise', 0),
                scores.get('disgust', 0),
                reply,
                datetime.utcnow()
            ))
    except Exception as db_error:
        print("❌ DB insert error:", db_error)

    return jsonify({
        "reply": reply,
        "audio": audio_url,
        "emotions": scores
    })

@app.route("/history", methods=["GET"])
def get_history():
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_message, bot_reply, joy, sadness, anger, fear, surprise, disgust, created_at
                FROM mood_logs
                ORDER BY created_at DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
            history = []
            for row in rows:
                history.append({
                    "user": row[0],
                    "bot": row[1],
                    "emotions": {
                        "joy": row[2],
                        "sadness": row[3],
                        "anger": row[4],
                        "fear": row[5],
                        "surprise": row[6],
                        "disgust": row[7]
                    },
                    "timestamp": row[8].isoformat()
                })
            return jsonify(history)
    except Exception as e:
        print("❌ Fetch history error:", e)
        return jsonify({"error": "Unable to fetch history"}), 500

if __name__ == "__main__":
    app.run(debug=True)
