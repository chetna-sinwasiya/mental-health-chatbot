# Step 1 - Install libraries
import subprocess
subprocess.run(["pip", "install", "textblob"], capture_output=True)

# Step 2 - Import sab kuch
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import random

print("Libraries loaded!")

# Step 3 - Training data (GoEmotions dataset)
import pandas as pd

df1 = pd.read_csv(r'C:\Users\chetn\Downloads\goemotion\data\full_dataset\goemotions_1.csv')
df2 = pd.read_csv(r'C:\Users\chetn\Downloads\goemotion\data\full_dataset\goemotions_2.csv')
df3 = pd.read_csv(r'C:\Users\chetn\Downloads\goemotion\data\full_dataset\goemotions_3.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

# Emotion column banao
emotion_cols = ['admiration','amusement','anger','annoyance','approval',
'caring','confusion','curiosity','desire','disappointment','disapproval',
'disgust','embarrassment','excitement','fear','gratitude','grief','joy',
'love','nervousness','optimism','pride','realization','relief','remorse',
'sadness','surprise','neutral']

df['emotion'] = df[emotion_cols].idxmax(axis=1)
df = df[['text', 'emotion']].dropna()

sentences = df['text'].tolist()
labels = df['emotion'].tolist()

print(f"Total training data: {len(sentences)} sentences!")
print("Data ready!")

# Step 4 - Naive Bayes model
nb_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
nb_model.fit(sentences, labels)
print("Naive Bayes ready!")

# Step 5 - LSTM model
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=20, padding='post')

label_map = {
    'admiration':0,'amusement':1,'anger':2,'annoyance':3,'approval':4,
    'caring':5,'confusion':6,'curiosity':7,'desire':8,'disappointment':9,
    'disapproval':10,'disgust':11,'embarrassment':12,'excitement':13,
    'fear':14,'gratitude':15,'grief':16,'joy':17,'love':18,
    'nervousness':19,'optimism':20,'pride':21,'realization':22,
    'relief':23,'remorse':24,'sadness':25,'surprise':26,'neutral':27
}
numeric_labels = [label_map[label] for label in labels]

model_lstm = Sequential([
    Embedding(1000, 16),
    LSTM(32),
    Dense(28, activation='softmax')
])
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(tf.constant(padded),
    tf.constant(numeric_labels),
    epochs=5,
    verbose=1)
print("LSTM ready!")

# Step 6 - Chatbot responses
def mental_health_chatbot(user_input):
    blob = TextBlob(user_input)
    score = blob.sentiment.polarity
    user_input_lower = user_input.lower()

    if "exam" in user_input_lower or "study" in user_input_lower:
        return random.choice([
            "Exam stress is real! Take one topic at a time. You've got this! 💪",
            "Break your syllabus into small parts. 30 min study, 10 min break!",
            "Remember — one exam doesn't define your future. Breathe! 🌸"
        ])
    elif "lonely" in user_input_lower or "alone" in user_input_lower or "akela" in user_input_lower:
        return random.choice([
            "You are never truly alone. I'm here with you! 💙",
            "Loneliness is hard. Try calling one friend today!",
            "It's okay to feel lonely sometimes. Be kind to yourself! 🤗"
        ])
    elif "anxious" in user_input_lower or "anxiety" in user_input_lower:
        return random.choice([
            "Try this: Breathe in 4 sec, hold 4, out 4. Repeat! 🌬️",
            "Anxiety is your mind overthinking. Focus on what you CAN control!",
            "Ground yourself — name 5 things you can see right now!"
        ])
    elif "sad" in user_input_lower or "depressed" in user_input_lower:
        return random.choice([
            "I hear you. Sadness is valid. You don't have to be okay all the time 💙",
            "Be gentle with yourself today. Small steps are still steps!",
            "If it gets too heavy, please talk to someone you trust. You matter! 🌸"
        ])
    elif "disturb" in user_input_lower or "ill" in user_input_lower or "mentally" in user_input_lower or "broken" in user_input_lower or "hopeless" in user_input_lower:
        return random.choice([
        "I hear you. That takes courage to say. You're not alone 💙",
        "Thank you for sharing this. Please consider talking to someone you trust 🌸",
        "You matter."
        ])
    elif score < -0.5:
        return random.choice([
            "That sounds really hard. I'm here for you! 💙",
            "You're stronger than you think. One breath at a time!",
            "It's okay to not be okay. Tomorrow can be better! 🌟"
        ])
    elif score < 0:
        return random.choice([
            "I hear you. Want to talk more about it?",
            "That sounds tough. Be kind to yourself today!",
            "You're doing better than you think! 😊"
        ])
    else:
        return random.choice([
            "That's wonderful! Keep that energy! 🌟",
            "Love to hear that! You're doing great!",
            "Amazing! Spread that positivity! ✨"
        ])

# Step 7 - LSTM predict function
def predict_emotion_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=20, padding='post')
    prediction = model_lstm.predict(padded_seq, verbose=0)
    emotion_map = {v:k for k,v in label_map.items()}
    predicted = np.argmax(prediction)
    confidence = round(np.max(prediction)*100, 2)
    return emotion_map[predicted], confidence

print("\n✅ Everything ready!")
print("="*40)
print("🤖 Mental Health Chatbot - AI Powered")
print("="*40)
print("Type 'quit' to exit\n")

# Step 8 - Conversation
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Take care! You're not alone! 💙")
        break
    emotion, confidence = predict_emotion_lstm(user_input)
    response = mental_health_chatbot(user_input)
    print(f"Chatbot: {response}")
    print(f"[AI: {emotion} | {confidence}% confident]\n")