"""
content_based.py
----------------
Content-Based Filtering using TF-IDF on genres + tags.
Computes cosine similarity between movies based on their metadata.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource(show_spinner=False)
def build_content_model(movies: pd.DataFrame):
    """
    Build TF-IDF matrix from genres and tags.
    Returns the similarity matrix and movie index mapping.
    """
    # Combine genres and tags into a single text feature
    movies = movies.copy()
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    movies["content_soup"] = movies["genres_clean"] + " " + movies["tags"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies["content_soup"].fillna(""))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Map movieId -> positional index in the matrix
    movie_idx = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    title_idx = pd.Series(movies.index, index=movies["clean_title"].str.lower()).to_dict()

    return cosine_sim, movie_idx, title_idx


def get_content_recommendations(
    movie_id: int,
    movies: pd.DataFrame,
    cosine_sim: np.ndarray,
    movie_idx: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top_n content-based recommendations for a given movieId."""
    if movie_id not in movie_idx:
        return pd.DataFrame()

    idx = movie_idx[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    movie_indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]

    recs = movies.iloc[movie_indices][
        ["movieId", "clean_title", "genres", "year", "avg_rating", "rating_count"]
    ].copy()
    recs["content_score"] = scores
    return recs.reset_index(drop=True)
