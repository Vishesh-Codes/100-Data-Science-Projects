# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function for preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Load the example dataset
data = pd.DataFrame({
    'text': ["good love like happy joyful excited lovely fantastic amazing excellent wonderful awesome great beautiful positive uplifting brilliant delightful vibrant success win triumph celebrate prosper blessed smile grateful pleasure satisfaction inspiring optimistic glorious radiant harmony praise kindness affection passionate compassionate elegant jubilant charming graceful fun victory splendid splendor exhilarating ecstatic fabulous heartwarming nice wow best", 
    "bad hate dislike sad unhappy disappointed horrible terrible awful negative stressful difficult annoying frustrating displeased disgusting dreadful unpleasant miserable upset regret angry irritating offensive boring ugly unfortunate unfavorable hopeless lonely depressing gloomy sorrowful tiring weary troubled grim distressed disheartened insulting injured harmful furious discontent disagreeable repulsive unfortunate dismal dismaying repugnant harsh despair desperate not"],
    'sentiment': ['positive', 'negative']
})

# Preprocess the example dataset
data['processed_text'] = data['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['sentiment']

# Train a classifier (Naive Bayes in this case)
classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

# Streamlit app
st.title("Sentiment Analysis App")

# User input for text
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess user input
        user_input_processed = preprocess_text(user_input)
        # Feature extraction using TF-IDF
        user_input_tfidf = tfidf_vectorizer.transform([user_input_processed])
        # Make prediction
        prediction = classifier.predict(user_input_tfidf)
        st.write(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter text for analysis.")
