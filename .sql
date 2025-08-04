CREATE TABLE mood_logs (
    id SERIAL PRIMARY KEY,
    user_message TEXT,
    joy FLOAT,
    sadness FLOAT,
    anger FLOAT,
    fear FLOAT,
    surprise FLOAT,
    disgust FLOAT,
    bot_reply TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
