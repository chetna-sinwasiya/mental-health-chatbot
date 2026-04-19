# 🧠 Mental Health Chatbot - AI Powered

An AI-powered mental health support chatbot built with Python that detects emotions from user input and provides empathetic, supportive responses using machine learning and deep learning.

---

## ✨ Features

- **Emotion Detection** — LSTM model classifies input into 28 emotion categories
- **Smart Responses** — Different responses for nervousness, sadness, anger, joy, love, and more
- **Keyword-based Detection** — Handles exam stress, motivation, loneliness, anxiety, depression
- **Crisis Detection** — Detects suicidal/harmful inputs and provides helpline numbers immediately
- **Hinglish Support** — Understands basic Hinglish inputs
- **Real-time Conversation** — Continuous chat with clean output

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Classification | Scikit-learn (Naive Bayes + TF-IDF) |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Data Handling | Pandas, NumPy |
| Dataset | GoEmotions (Google) |

---

## 🧬 Models Used

### 1. Naive Bayes (Scikit-learn)
TF-IDF vectorizer converts text to numerical features. MultinomialNB classifies input into one of 28 emotion labels based on probability.

### 2. LSTM (TensorFlow/Keras)
A sequential deep learning model that captures word sequence context for better emotion prediction. Trained on the GoEmotions dataset.

---

## 📁 Dataset

**GoEmotions** by Google — 211,225 Reddit comments labeled across 28 emotion categories:

`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral`

Download: [GoEmotions on GitHub](https://github.com/google-research/google-research/tree/master/goemotions)

---

## 📂 Project Structure

```
mental-health-chatbot/
│
├── chatbot.py              # Main chatbot script
├── README.md               # Project documentation
│
└── data/
    └── full_dataset/
        ├── goemotions_1.csv
        ├── goemotions_2.csv
        └── goemotions_3.csv
```

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/chetna-sinwasiya/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install dependencies
```bash
pip install textblob scikit-learn tensorflow numpy pandas
```

### 3. Download the GoEmotions dataset
Steps to download:
1. Go to GoEmotions Dataset
2. Download these 3 files:
   - goemotions_1.csv
   - goemotions_2.csv
   - goemotions_3.csv
Place the CSV files inside `data/full_dataset/` as shown in the project structure above.

### 4. Run the chatbot
```bash
python chatbot.py
```

---

## 💬 Sample Chat

```
🤖 Mental Health Chatbot - AI Powered
========================================
Type 'quit' to exit

You: i feel so nervous about my exam
Chatbot: Nervousness means you care! You've got this! 💪

You: i have been feeling really sad lately
Chatbot: I hear you. Sadness is valid. You don't have to be okay all the time 💙

You: please motivate me
Chatbot: Every day is a new chance to be better! You've got this! 🌟

You: i want to die
Chatbot: I'm really concerned about you. Please call iCall: 9152987821 💙

You: bye
Chatbot: Take care! You're not alone! 💙
```

---

## ⚙️ How It Works

```
User Input
    ↓
Keyword Matching (motivate, exam, disturb, crisis keywords...)
    ↓ (if no keyword match)
LSTM Emotion Prediction (28 emotion classes)
    ↓
Response Generated based on detected emotion
```

---

## 🆘 Crisis Detection

If a user types anything related to self-harm or suicide, the chatbot immediately provides helpline numbers:

- 📞 **iCall** — 9152987821 (Tata Institute of Social Sciences)
- 📞 **Vandrevala Foundation** — 1860-2662-345 (24/7 Available)
