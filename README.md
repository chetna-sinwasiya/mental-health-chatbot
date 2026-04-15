# Mental Health Chatbot
An AI-based chatbot developed using Python that analyzes user input to detect emotions and provide supportive responses. It combines NLP, machine learning, and deep learning techniques to improve accuracy and interaction.

## Features
- Emotion detection using sentiment analysis (TextBlob) with polarity scoring
- Machine learning-based classification using Naive Bayes
- Deep learning model (LSTM) for improved accuracy and context-aware emotion prediction
- Keyword-based response system for better interaction
- Crisis detection mechanism to identify sensitive inputs and provide helpline support
- Real-time conversational interaction with continuous user input handling
- Support for varied inputs including basic Hinglish through custom training data 

## How It Works
1. User inputs text
2. Sentiment analysis generates a polarity score
3. ML/DL models classify emotion
4. Chatbot generates appropriate response
5. Crisis keywords trigger helpline suggestions

## Tech Stack
- Python
- TextBlob (NLP, Sentiment Analysis)
- Scikit-learn (Naive Bayes)
- TensorFlow/Keras (LSTM)
- NumPy
- Google Colab

## Models Used
- **TextBlob**: Performs sentiment analysis and assigns polarity scores (-1 to +1)
- **Naive Bayes**: Classifies text into emotion categories using probability
- **LSTM**: Deep learning model that captures sequence context for better predictions (achieved ~90% accuracy)


