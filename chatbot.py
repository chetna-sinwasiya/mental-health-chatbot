
# Step 1 - Import everything
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
import pandas as pd

print("Libraries loaded!")

# Step 2 - Training data (GoEmotions dataset)
df1 = pd.read_csv('data/full_dataset/goemotions_1.csv')
df2 = pd.read_csv('data/full_dataset/goemotions_2.csv')
df3 = pd.read_csv('data/full_dataset/goemotions_3.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

# Make emotion column 
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

# Step 3 - Naive Bayes model
nb_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
nb_model.fit(sentences, labels)
print("Naive Bayes ready!")

# Step 4 - LSTM model
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=30, padding='post')

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
    Embedding(5000, 16),
    LSTM(32),
    Dense(28, activation='softmax')
])
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(tf.constant(padded),
    tf.constant(numeric_labels),
    epochs=3,
    verbose=1)
print("LSTM ready!")

# Step 5 - Chatbot responses
def mental_health_chatbot(user_input, emotion):
    user_input_lower = user_input.lower()
     if "motivate" in user_input_lower or "motivation" in user_input_lower or "inspire" in user_input_lower:
        return random.choice([
            "You are capable of amazing things! Keep going! 💪",
            "Every day is a new chance to be better! You've got this! 🌟",
            "Believe in yourself — you're stronger than you think! ✨"
        ])
    elif "exam" in user_input_lower or "study" in user_input_lower:
        return random.choice([
            "Exam stress is real! Take one topic at a time. You've got this! 💪",
            "Break your syllabus into small parts. 30 min study, 10 min break!",
            "Remember — one exam doesn't define your future. Breathe! 🌸"
        ])
    elif emotion in ["nervousness", "fear", "anxiety"]:
        return random.choice([
            "It's okay to feel nervous. Take a deep breath! 🌬️",
            "Nervousness means you care! You've got this! 💪",
            "Feel the fear and do it anyway. You're braver than you think! 🌸"
        ])
    elif emotion in ["sadness", "grief", "remorse", "disappointment"] or "sad" in user_input_lower or "depressed" in user_input_lower:
        return random.choice([
            "I hear you. Sadness is valid. You don't have to be okay all the time 💙",
            "Be gentle with yourself today. Small steps are still steps!",
            "If it gets too heavy, please talk to someone you trust. You matter! 🌸"
        ])
    elif emotion in ["anger", "annoyance", "disapproval", "disgust"]:
        return random.choice([
            "It's okay to feel angry. Take a moment to breathe! 🌬️",
            "Your feelings are valid. Try to channel that energy positively!",
            "Anger is natural. Be kind to yourself! 💙"
        ])
    elif emotion in ["joy", "amusement", "excitement", "optimism", "pride"]:
        return random.choice([
            "That's wonderful! Keep that energy! 🌟",
            "Love to hear that! You're doing great!",
            "Amazing! Spread that positivity! ✨"
        ])
    elif emotion in ["love", "caring", "gratitude", "admiration"]:
        return random.choice([
            "That's beautiful! Hold onto that feeling! 💙",
            "Love and gratitude make life better! 🌸",
            "You have such a kind heart! 🤗"
        ])
    elif emotion in ["confusion", "realization", "surprise"]:
        return random.choice([
            "It's okay to feel confused sometimes. Take it one step at a time!",
            "Life can be surprising! Embrace the uncertainty! 🌟",
            "Confusion means you're thinking deeply. That's great! 😊"
        ])
         elif "disturb" in user_input_lower or "ill" in user_input_lower or "mentally" in user_input_lower or "broken" in user_input_lower or "hopeless" in user_input_lower:
        return random.choice([
            "I hear you. That takes courage to say. You're not alone 💙",
            "Thank you for sharing this. Please consider talking to someone you trust 🌸",
            "You matter. Seeking help is a sign of strength, not weakness 💙"
        ])
    elif any(word in user_input_lower for word in ["suicide", "suscide", "want to die", "kill myself", "end my life", "not want to live", "don't want to live"]):
        return random.choice([
            "I'm really concerned about you. Please call iCall: 9152987821 💙",
            "You matter so much. Please reach out: Vandrevala Foundation: 1860-2662-345 💙",
            "Please don't give up. You are not alone. Call iCall: 9152987821 🌸"
        ])
    else:
        return random.choice([
            "I'm here for you! Tell me more 💙",
            "You're not alone. I'm listening! 🤗",
            "Thank you for sharing. How can I help? 🌸"
        ])

# Step 6 - LSTM predict function
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

# Step 7 - Conversation
while True:
    user_input = input("You: ")
     if user_input.lower() in ["quit", "bye", "exit", "goodbye"]:
        print("Chatbot: Take care! You're not alone! 💙")
        break
    emotion, confidence = predict_emotion_lstm(user_input)
    response = mental_health_chatbot(user_input, emotion)
    print(f"Chatbot: {response}")
    print()
