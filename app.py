import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure NLTK data is available
nltk.data.path.append('./nltk_data')
os.makedirs('./nltk_data', exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    """Preprocess the input text by lowering case, tokenizing, removing stopwords/punctuation, and stemming."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    filtered_tokens = [token for token in tokens if token.isalnum()]

    # Remove stopwords and punctuation
    filtered_tokens = [token for token in filtered_tokens if token not in stop_words and token not in string.punctuation]

    # Stem tokens
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

    return " ".join(stemmed_tokens)

# Load model and vectorizer with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the working directory.")
    st.stop()

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS or Email text")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # Preprocess input text
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Show result
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
