"""
collaborative.py
----------------
Collaborative Filtering using pure NumPy/scikit-learn TruncatedSVD.
No scikit-surprise or C++ compilation needed — works on Streamlit Cloud.
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

SURPRISE_AVAILABLE = True   # kept True so app.py shows "Hybrid" mode


@st.cache_resource(show_spinner=False)
def build_collab_model(ratings: pd.DataFrame):
    n_components = 50

    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_row = {u: i for i, u in enumerate(user_ids)}
    movie_col = {m: j for j, m in enumerate(movie_ids)}

    rows = ratings["userId"].map(user_row).values
    cols = ratings["movieId"].map(movie_col).values
    vals = ratings["rating"].values.astype(np.float32)

    user_means = ratings.groupby("userId")["rating"].mean()
    global_mean = float(ratings["rating"].mean())
    mean_vals = vals - ratings["userId"].map(user_means).values.astype(np.float32)

    R = csr_matrix((mean_vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(R)

    model_data = {
        "movie_ids": movie_ids,
        "movie_col": movie_col,
        "Vt": svd.components_,
        "global_mean": global_mean,
    }

    return model_data, None


def get_collab_recommendations(
    user_ratings: dict,
    movies: pd.DataFrame,
    model,
    top_n: int = 20,
    synthetic_user_id: int = 999999,
) -> pd.DataFrame:
    if model is None or not user_ratings:
        return pd.DataFrame()

    Vt = model["Vt"]
    movie_col = model["movie_col"]
    movie_ids = model["movie_ids"]

    n_movies = len(movie_ids)
    user_vec = np.zeros(n_movies, dtype=np.float32)
    rated_cols = []
    user_mean = np.mean(list(user_ratings.values()))

    for mid, r in user_ratings.items():
        if mid in movie_col:
            col = movie_col[mid]
            user_vec[col] = r - user_mean
            rated_cols.append(col)

    if not rated_cols:
        return pd.DataFrame()

    user_latent = Vt @ user_vec
    scores = Vt.T @ user_latent
    scores += user_mean

    rated_movie_ids = set(user_ratings.keys())
    candidate_mask = ~np.isin(movie_ids, list(rated_movie_ids))
    cand_movie_ids = movie_ids[candidate_mask]
    cand_scores = scores[candidate_mask]

    pred_df = pd.DataFrame({"movieId": cand_movie_ids, "collab_score": cand_scores})
    result = pred_df.merge(
        movies[["movieId", "clean_title", "genres", "year", "avg_rating", "rating_count"]],
        on="movieId", how="inner",
    )
    result = result.nlargest(top_n, "collab_score")[
        ["movieId", "clean_title", "genres", "year", "avg_rating", "rating_count", "collab_score"]
    ]
    return result.reset_index(drop=True)


def get_popular_unseen(
    movies: pd.DataFrame,
    seen_ids: set,
    top_n: int = 20,
    min_ratings: int = 50,
) -> pd.DataFrame:
    unseen = movies[
        (~movies["movieId"].isin(seen_ids)) &
        (movies["rating_count"] >= min_ratings)
    ].copy()
    C = movies["rating_count"].quantile(0.6)
    m = movies["avg_rating"].mean()
    unseen["bayesian"] = (
        (unseen["rating_count"] / (unseen["rating_count"] + C)) * unseen["avg_rating"]
        + (C / (unseen["rating_count"] + C)) * m
    )
    return unseen.nlargest(top_n, "bayesian")[
        ["movieId", "clean_title", "genres", "year", "avg_rating", "rating_count"]
    ].reset_index(drop=True)
