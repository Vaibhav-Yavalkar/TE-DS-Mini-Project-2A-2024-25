import re
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import requests

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to clean text
def clean_text(text):
    
    text = re.sub(r'@[\w]*', '', text)  # Remove mentions
    text = re.sub('[^a-zA-Z#]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize words
    return text

# Load the model and tokenizer
def load_resources():
    try:
        model = load_model('LSTM_emotion_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Function to predict the emotion from the text
def detect_emotion(text, model, tokenizer, sequence_length=70):
    
    cleaned_text = clean_text(text)
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(text_sequence, maxlen=sequence_length)
    
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=-1)
    
    # Define the label mapping based on your training labels
    labels = {0: "Anger", 1: "Fear", 2: "Joy", 3: "Neutral", 4: "Sad"}
    return labels[predicted_label[0]], prediction
     
    

# Main function to run the Streamlit app
def main():
    st.title("Track your mOOd")

    # Load model and tokenizer
    model, tokenizer = load_resources()
    if not model or not tokenizer:
        return

    # Get user input
    input_text = st.text_area("Enter a text:")

    # Classify the mood when the button is pressed
    if st.button("Classify"):
        if not input_text.strip():
            st.write("Please enter some text.")
        else:
            # Predict the emotion
            emotion, prediction = detect_emotion(input_text, model, tokenizer)

            # Display the result
            st.write(f"Detected Emotion: {emotion}")

            # Optional: For debugging, show intermediate results
            st.write(f"Prediction Raw Output: {prediction}")

# Run the app
if __name__ == "__main__":
    main()
