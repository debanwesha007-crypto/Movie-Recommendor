"""
data_loader.py
--------------
Downloads and preprocesses the MovieLens Small dataset.
Provides movies DataFrame and ratings DataFrame ready for use.
"""

import os
import zipfile
import requests
import pandas as pd
import streamlit as st

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "ml-latest-small"


def download_data():
    """Download MovieLens small dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        with st.spinner("📥 Downloading MovieLens dataset..."):
            response = requests.get(MOVIELENS_URL, stream=True)
            zip_path = "ml-latest-small.zip"
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(".")
            os.remove(zip_path)


@st.cache_data(show_spinner=False)
def load_data():
    """Load and return movies and ratings DataFrames."""
    download_data()

    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))

    # Extract year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    # Merge tags into movies
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
        .rename(columns={"tag": "tags"})
    )
    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tags"] = movies["tags"].fillna("")

    # Average rating & count per movie
    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    movies = movies.merge(rating_stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0).round(2)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)

    return movies, ratings
