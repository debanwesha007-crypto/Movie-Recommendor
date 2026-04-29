"""
app.py
------
Main Streamlit application for the Hybrid Movie Recommender System.
Combines Content-Based + Collaborative Filtering.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0d14;
    color: #e8e4dc;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12121c;
    border-right: 1px solid #2a2a3d;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    background: linear-gradient(135deg, #1a0a2e 0%, #0d0d14 60%);
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #2a1a4a;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(180,100,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #c084fc, #f59e0b, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Movie Card ── */
.movie-card {
    background: #16162a;
    border: 1px solid #2a2a3d;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
}
.movie-card:hover {
    border-color: #c084fc;
    transform: translateY(-2px);
}
.movie-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #2d2d45;
    position: absolute;
    top: 0.8rem;
    right: 1.1rem;
}
.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8e4dc;
    margin: 0 0 0.2rem;
}
.movie-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.genre-chip {
    display: inline-block;
    background: #1e1e35;
    border: 1px solid #3a3a55;
    border-radius: 20px;
    padding: 0.15rem 0.55rem;
    font-size: 0.7rem;
    color: #a78bfa;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
}
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.6rem;
}
.score-label {
    font-size: 0.68rem;
    color: #6b7280;
    width: 56px;
    text-align: right;
    flex-shrink: 0;
}
.score-bar-bg {
    flex: 1;
    height: 5px;
    background: #1e1e35;
    border-radius: 3px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.score-val {
    font-size: 0.68rem;
    color: #9ca3af;
    width: 34px;
    text-align: right;
}

/* ── Stat box ── */
.stat-box {
    background: #16162a;
    border: 1px solid #2a2a3d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #c084fc;
}
.stat-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Section title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e4dc;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a2a3d;
}

/* ── Alpha slider label ── */
.alpha-label {
    font-size: 0.75rem;
    color: #9ca3af;
}

/* ── Star rating ── */
.stars { color: #f59e0b; font-size: 0.9rem; }

/* ── Streamlit overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #9ca3af !important;
    font-size: 0.8rem !important;
}
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #c084fc);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    cursor: pointer;
    width: 100%;
}
.stButton>button:hover {
    opacity: 0.9;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Load data & models ────────────────────────────────────────────────────────
from data_loader import load_data
from content_based import build_content_model
from collaborative import build_collab_model, SURPRISE_AVAILABLE
from hybrid import hybrid_recommend


@st.cache_resource(show_spinner=False)
def load_all():
    movies, ratings = load_data()
    cosine_sim, movie_idx, title_idx = build_content_model(movies)
    collab_model, _ = build_collab_model(ratings)
    return movies, ratings, cosine_sim, movie_idx, title_idx, collab_model


with st.spinner("🎬 Loading CineMatch engine…"):
    movies, ratings, cosine_sim, movie_idx, title_idx, collab_model = load_all()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <p class="hero-title">CineMatch</p>
    <p class="hero-sub">Hybrid · Content × Collaborative · MovieLens Powered</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="stat-box"><div class="stat-value">{len(movies):,}</div><div class="stat-label">Movies</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="stat-box"><div class="stat-value">{len(ratings):,}</div><div class="stat-label">Ratings</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    n_users = ratings["userId"].nunique()
    st.markdown(
        f'<div class="stat-box"><div class="stat-value">{n_users:,}</div><div class="stat-label">Users</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    mode = "Hybrid" if (SURPRISE_AVAILABLE and collab_model) else "Content-Only"
    st.markdown(
        f'<div class="stat-box"><div class="stat-value" style="font-size:1.3rem">{mode}</div><div class="stat-label">Active Mode</div></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Controls")

    # Movie selector
    movie_titles = sorted(movies["clean_title"].dropna().unique().tolist())
    selected_title = st.selectbox("🎬 Pick a seed movie", movie_titles, index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0)

    seed_row = movies[movies["clean_title"] == selected_title].iloc[0]
    seed_id = int(seed_row["movieId"])

    # Number of recommendations
    top_n = st.slider("🎯 Recommendations", min_value=5, max_value=20, value=10, step=1)

    # Alpha slider
    if SURPRISE_AVAILABLE and collab_model:
        st.markdown("---")
        st.markdown("**⚖️ Blend Weight**")
        alpha = st.slider(
            "Content ←→ Collaborative",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0 = pure collaborative, 1 = pure content-based"
        )
        st.markdown(
            f'<p class="alpha-label">Content: {alpha:.0%} · Collaborative: {1-alpha:.0%}</p>',
            unsafe_allow_html=True,
        )
    else:
        alpha = 0.7
        st.info("💡 Install `scikit-surprise` to enable full collaborative filtering.")

    # Optional: user ratings input
    st.markdown("---")
    st.markdown("**⭐ Rate movies you've seen** *(boosts collaborative signal)*")

    genre_list = sorted(
        set(g for genres in movies["genres"].dropna() for g in genres.split("|") if g != "(no genres listed)")
    )
    fav_genre = st.selectbox("Filter by genre", ["All"] + genre_list)

    filtered = movies if fav_genre == "All" else movies[movies["genres"].str.contains(fav_genre, na=False)]
    rate_titles = sorted(filtered["clean_title"].dropna().unique().tolist())

    user_ratings = {}
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    rate_movie = st.selectbox("Choose a movie to rate", rate_titles)
    rating_val = st.slider("Your rating", 0.5, 5.0, 3.5, 0.5)

    if st.button("Add Rating"):
        row = movies[movies["clean_title"] == rate_movie]
        if not row.empty:
            mid = int(row.iloc[0]["movieId"])
            st.session_state.user_ratings[mid] = rating_val
            st.success(f"Rated '{rate_movie}' → {rating_val} ⭐")

    if st.session_state.user_ratings:
        st.markdown(f"*{len(st.session_state.user_ratings)} movie(s) rated*")
        if st.button("Clear Ratings"):
            st.session_state.user_ratings = {}

    user_ratings = st.session_state.user_ratings


# ── Seed Movie Info ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎞 Seed Movie</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 1])
with col_a:
    genres_display = seed_row["genres"].replace("|", " · ") if pd.notna(seed_row["genres"]) else "N/A"
    year_display = str(int(seed_row["year"])) if pd.notna(seed_row["year"]) else "N/A"
    st.markdown(
        f"""
<div class="movie-card">
    <div class="movie-title">{seed_row['clean_title']}</div>
    <div class="movie-meta">📅 {year_display} &nbsp;|&nbsp; 🎭 {genres_display}</div>
    <div class="movie-meta">⭐ {seed_row['avg_rating']:.2f} avg · {seed_row['rating_count']:,} ratings</div>
</div>
""",
        unsafe_allow_html=True,
    )
with col_b:
    st.metric("Avg Rating", f"{seed_row['avg_rating']:.2f} ⭐")
    st.metric("Total Ratings", f"{int(seed_row['rating_count']):,}")


# ── Run Hybrid Recommender ────────────────────────────────────────────────────
with st.spinner("🔮 Computing hybrid recommendations…"):
    recs = hybrid_recommend(
        seed_movie_id=seed_id,
        user_ratings=user_ratings,
        movies=movies,
        cosine_sim=cosine_sim,
        movie_idx=movie_idx,
        collab_model=collab_model if SURPRISE_AVAILABLE else None,
        top_n=top_n,
        alpha=alpha,
    )


# ── Display Results ───────────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-title">🍿 Top {top_n} Recommendations for "{selected_title}"</div>',
    unsafe_allow_html=True,
)

if recs.empty:
    st.warning("No recommendations found. Try a different movie.")
else:
    col_left, col_right = st.columns(2)

    for i, (_, row) in enumerate(recs.iterrows()):
        col = col_left if i % 2 == 0 else col_right

        genres_chips = "".join(
            f'<span class="genre-chip">{g.strip()}</span>'
            for g in str(row.get("genres", "")).split("|")
            if g.strip() and g.strip() != "(no genres listed)"
        )

        year_str = str(int(row["year"])) if pd.notna(row.get("year")) else "—"
        avg_r = row.get("avg_rating", 0)
        c_score = row.get("content_score", 0)
        cf_score = row.get("collab_score", 0)
        h_score = row.get("hybrid_score", 0)

        def bar(val, color):
            pct = int(val * 100)
            return (
                f'<div class="score-bar-bg">'
                f'<div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>'
                f'</div>'
            )

        with col:
            st.markdown(
                f"""
<div class="movie-card">
    <span class="movie-rank">#{i+1}</span>
    <div class="movie-title">{row['clean_title']}</div>
    <div class="movie-meta">📅 {year_str} &nbsp;|&nbsp; ⭐ {avg_r:.2f} ({int(row.get('rating_count',0)):,} ratings)</div>
    <div>{genres_chips}</div>
    <div class="score-bar-wrap">
        <span class="score-label">Content</span>
        {bar(c_score, '#a78bfa')}
        <span class="score-val">{c_score:.2f}</span>
    </div>
    <div class="score-bar-wrap">
        <span class="score-label">Collab</span>
        {bar(cf_score, '#34d399')}
        <span class="score-val">{cf_score:.2f}</span>
    </div>
    <div class="score-bar-wrap">
        <span class="score-label">Hybrid</span>
        {bar(h_score, '#f59e0b')}
        <span class="score-val">{h_score:.2f}</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ── Score distribution chart ──────────────────────────────────────────────────
if not recs.empty:
    st.markdown('<div class="section-title">📊 Score Distribution</div>', unsafe_allow_html=True)
    chart_data = recs[["clean_title", "content_score", "collab_score", "hybrid_score"]].set_index("clean_title")
    st.bar_chart(chart_data, height=320, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4b5563;font-size:0.75rem;">CineMatch · Built with MovieLens Small Dataset · Powered by scikit-learn & Streamlit</p>',
    unsafe_allow_html=True,
)
