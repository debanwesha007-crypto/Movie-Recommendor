"""
hybrid.py
---------
Hybrid Recommender: blends Content-Based and Collaborative scores
using a weighted combination. Falls back to content-only if no user
ratings are provided.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def hybrid_recommend(
    seed_movie_id: int,
    user_ratings: dict,           # {movieId: rating} — can be empty {}
    movies: pd.DataFrame,
    cosine_sim: np.ndarray,
    movie_idx: dict,
    collab_model,
    top_n: int = 10,
    alpha: float = 0.5,          # weight for content (1-alpha = collab weight)
) -> pd.DataFrame:
    """
    Hybrid recommendation combining content-based and collaborative filtering.

    Parameters
    ----------
    seed_movie_id : int
        The anchor movie the user selected.
    user_ratings : dict
        User-provided {movieId: rating} pairs (can be empty).
    movies : pd.DataFrame
        Full movies DataFrame.
    cosine_sim : np.ndarray
        Precomputed cosine similarity matrix.
    movie_idx : dict
        movieId -> matrix row index mapping.
    collab_model : SVD model or None
        Trained collaborative filtering model.
    top_n : int
        Number of final recommendations.
    alpha : float
        Weight given to content-based score (0–1).

    Returns
    -------
    pd.DataFrame with columns:
        movieId, clean_title, genres, year, avg_rating,
        rating_count, content_score, collab_score, hybrid_score
    """
    from content_based import get_content_recommendations
    from collaborative import get_collab_recommendations, get_popular_unseen

    # --- Content-Based Candidates ---
    content_recs = get_content_recommendations(
        seed_movie_id, movies, cosine_sim, movie_idx, top_n=50
    )

    if content_recs.empty:
        return pd.DataFrame()

    # --- Collaborative Candidates ---
    if collab_model is not None and user_ratings:
        collab_recs = get_collab_recommendations(
            user_ratings, movies, collab_model, top_n=50
        )
    else:
        # Fall back to popularity-weighted scores
        seen_ids = set(user_ratings.keys()) | {seed_movie_id}
        collab_recs = get_popular_unseen(movies, seen_ids, top_n=50)
        collab_recs["collab_score"] = (
            collab_recs["avg_rating"] / 5.0
        )  # normalise 0–1

    # --- Merge on movieId ---
    merged = content_recs.merge(
        collab_recs[["movieId", "collab_score"]],
        on="movieId",
        how="outer",
    )
    merged = merged.merge(
        movies[["movieId", "clean_title", "genres", "year", "avg_rating", "rating_count"]],
        on="movieId",
        how="left",
        suffixes=("", "_movies"),
    )

    # Fill clean_title/genres from movies table where missing
    for col in ["clean_title", "genres", "year", "avg_rating", "rating_count"]:
        alt = col + "_movies"
        if alt in merged.columns:
            merged[col] = merged[col].combine_first(merged[alt])
            merged.drop(columns=[alt], inplace=True)

    # Drop the seed movie itself
    merged = merged[merged["movieId"] != seed_movie_id].copy()

    # Normalise both scores to [0, 1]
    scaler = MinMaxScaler()
    for score_col in ["content_score", "collab_score"]:
        if score_col not in merged.columns:
            merged[score_col] = 0.0
        merged[score_col] = merged[score_col].fillna(0.0)
        vals = merged[[score_col]].values
        if vals.max() > vals.min():
            merged[score_col] = scaler.fit_transform(vals).flatten()

    # Weighted hybrid score
    merged["hybrid_score"] = (
        alpha * merged["content_score"] + (1 - alpha) * merged["collab_score"]
    )

    result = (
        merged.nlargest(top_n, "hybrid_score")[
            [
                "movieId", "clean_title", "genres", "year",
                "avg_rating", "rating_count",
                "content_score", "collab_score", "hybrid_score",
            ]
        ]
        .reset_index(drop=True)
    )
    return result
