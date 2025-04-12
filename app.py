# README.md

"""
🎬 Movie Recommendation App

A simple content-based movie recommender system built using Python and Streamlit. The app suggests similar movies based on a user's selected favorite, using genre and keyword similarities from the TMDB 5000 Movie Dataset.

---

📌 Project Overview

This project uses basic natural language processing and cosine similarity to compare movie overviews, genres, and keywords. It's designed for quick deployment and interactive use through Streamlit.

---

🛠️ Tech Stack

- Python (pandas, scikit-learn)
- Streamlit (for the app interface)
- Cosine Similarity (for recommendations)
- Dataset: TMDB 5000 Movie Dataset on Kaggle

---

🚀 Features

- Select a favorite movie and get top 5 similar suggestions
- Interactive search with dropdown
- Genre and keyword matching
- Simple and clean Streamlit UI

---

📷 Demo Screenshot

*(Insert screenshot of your app here)*
> e.g., ![App Screenshot](screenshot.png)

---

🔧 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/movie-recommender

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

📁 Project Structure

```
movie-recommender/
│
├── app.py                  # Main Streamlit app
├── movies.csv              # Cleaned movie data
├── recommender.py          # Logic for similarity and matching
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

📈 Future Enhancements

- Add IMDb or TMDB API integration for movie posters
- Allow filtering by language or year
- Add collaborative filtering features (optional)

---

👨‍💻 Author

**Bardan Dahal** – Machine Learning Analyst  
NorQuest College, Edmonton
"""

# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'overview']].dropna()
    return df

def recommend(movie_title, df, cosine_sim):
    index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Main
st.title("🎬 Movie Recommendation App")
df = load_data()

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

selected_movie = st.selectbox("Choose a movie:", df['title'].values)

if st.button("Recommend"):
    try:
        recommendations = recommend(selected_movie, df, cosine_sim)
        st.subheader("You might also like:")
        for title in recommendations:
            st.write(f"- {title}")
    except IndexError:
        st.warning("Movie not found. Try another title.")
