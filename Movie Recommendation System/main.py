import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movie data from CSV file
movies_df = pd.read_csv('movies.csv')

# Check for and handle null values
movies_df = movies_df.fillna('')  # Replace null values with an empty string

# Preprocess the data
movies_df['Features'] = movies_df['genre'] + ' ' + movies_df['cast']

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title):
    idx = movies_df.index[movies_df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the input movie itself and get top 10

    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Streamlit app
st.title("Movie Recommendation App")

# User input for movie name
user_input = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    # Get recommendations
    recommendations = get_recommendations(user_input)

    # Display recommendations
    st.subheader(f"Top 10 movie recommendations for {user_input}:")
    st.write(recommendations)
