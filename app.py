"""
app.py — CineMatch v2  (Interactive Redesign)
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg:       #080810;
  --surface:  #0f0f1a;
  --card:     #13131f;
  --border:   #1e1e30;
  --accent:   #e8b14a;
  --accent2:  #6c63ff;
  --accent3:  #ff6584;
  --text:     #ddd8cc;
  --muted:    #555570;
  --radius:   14px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
  color: var(--text);
}

.stApp { background: var(--bg); }

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 3rem !important; max-width: 1400px !important; }

/* ── NAVBAR ─────────────────────────────────────────────────── */
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.2rem 0 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.nav-logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.2rem; letter-spacing: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.nav-tagline { font-size: .72rem; color: var(--muted); letter-spacing: .15em; text-transform: uppercase; margin-top: -4px; }
.nav-pills { display: flex; gap: .5rem; }
.nav-pill {
  padding: .35rem .9rem; border-radius: 20px; font-size: .75rem; font-weight: 500;
  border: 1px solid var(--border); color: var(--muted); background: transparent; cursor: pointer;
  transition: all .2s;
}
.nav-pill.active, .nav-pill:hover {
  border-color: var(--accent); color: var(--accent); background: rgba(232,177,74,.08);
}

/* ── SEARCH BAR ─────────────────────────────────────────────── */
.search-wrap {
  position: relative; margin-bottom: 1.5rem;
}
.search-icon {
  position: absolute; left: 1rem; top: 50%; transform: translateY(-50%);
  font-size: 1rem; color: var(--muted); pointer-events: none;
}
div[data-testid="stTextInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 40px !important;
  color: var(--text) !important;
  padding: .75rem 1.2rem .75rem 2.8rem !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: .95rem !important;
  transition: border-color .2s !important;
  width: 100% !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(232,177,74,.1) !important;
  outline: none !important;
}

/* ── GENRE CHIPS ─────────────────────────────────────────────── */
.genre-row { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 1.5rem; }
.gchip {
  padding: .3rem .85rem; border-radius: 20px; font-size: .72rem; font-weight: 500;
  border: 1px solid var(--border); color: var(--muted); cursor: pointer;
  transition: all .18s; white-space: nowrap;
}
.gchip.active {
  border-color: var(--accent2); color: #fff;
  background: linear-gradient(135deg, var(--accent2), #9f97ff);
  box-shadow: 0 2px 12px rgba(108,99,255,.35);
}
.gchip:hover:not(.active) { border-color: var(--accent2); color: var(--accent2); }

/* ── BLEND SLIDER DISPLAY ────────────────────────────────────── */
.blend-display {
  display: flex; align-items: center; gap: 1rem;
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem 1.4rem; margin-bottom: 1.5rem;
}
.blend-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.blend-bar-track {
  flex: 1; height: 6px; border-radius: 3px;
  background: linear-gradient(90deg, var(--accent2), var(--accent3));
  position: relative;
}
.blend-dot {
  position: absolute; top: 50%; width: 14px; height: 14px;
  border-radius: 50%; background: #fff; transform: translateY(-50%);
  box-shadow: 0 2px 8px rgba(0,0,0,.5);
  transition: left .2s;
}
.blend-pct { font-size: .78rem; font-weight: 600; color: var(--text); min-width: 36px; text-align: right; }

/* ── STATS ROW ───────────────────────────────────────────────── */
.stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-card {
  flex: 1; background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem 1.2rem;
  display: flex; flex-direction: column; align-items: flex-start;
  animation: fadeUp .5s ease both;
}
.stat-card:nth-child(1) { animation-delay: .05s }
.stat-card:nth-child(2) { animation-delay: .1s }
.stat-card:nth-child(3) { animation-delay: .15s }
.stat-card:nth-child(4) { animation-delay: .2s }
.stat-num {
  font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; line-height: 1;
  color: var(--accent); letter-spacing: 1px;
}
.stat-lbl { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-top: .2rem; }

/* ── SEED MOVIE BANNER ───────────────────────────────────────── */
.seed-banner {
  background: linear-gradient(135deg, #1a1428 0%, #0f0f1a 100%);
  border: 1px solid #2d2240;
  border-radius: var(--radius); padding: 1.4rem 1.6rem;
  display: flex; align-items: center; gap: 1.5rem;
  margin-bottom: 2rem; position: relative; overflow: hidden;
  animation: fadeUp .4s ease both;
}
.seed-banner::before {
  content: ''; position: absolute; right: -40px; top: -40px;
  width: 160px; height: 160px; border-radius: 50%;
  background: radial-gradient(circle, rgba(108,99,255,.15) 0%, transparent 70%);
}
.seed-poster {
  width: 54px; height: 80px; border-radius: 8px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent2), var(--accent3));
  display: flex; align-items: center; justify-content: center;
  font-size: 1.6rem;
}
.seed-info { flex: 1; }
.seed-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; letter-spacing: 1px; color: #fff; }
.seed-sub { font-size: .78rem; color: var(--muted); margin-top: .2rem; }
.seed-genres { margin-top: .5rem; display: flex; flex-wrap: wrap; gap: .3rem; }
.seed-genre-chip {
  padding: .15rem .55rem; border-radius: 12px; font-size: .65rem;
  background: rgba(108,99,255,.15); border: 1px solid rgba(108,99,255,.3); color: #9f97ff;
}
.seed-stats { text-align: right; flex-shrink: 0; }
.seed-rating { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; color: var(--accent); line-height: 1; }
.seed-rcount { font-size: .7rem; color: var(--muted); }

/* ── MOVIE GRID ──────────────────────────────────────────────── */
.rec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.rec-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.2rem;
  position: relative; cursor: pointer;
  transition: transform .2s, border-color .2s, box-shadow .2s;
  animation: fadeUp .5s ease both;
}
.rec-card:hover {
  transform: translateY(-4px);
  border-color: var(--accent);
  box-shadow: 0 8px 32px rgba(232,177,74,.12);
}
.rec-rank {
  position: absolute; top: .9rem; right: .9rem;
  font-family: 'Bebas Neue', sans-serif; font-size: 2rem;
  color: rgba(255,255,255,.06); line-height: 1;
}
.rec-accent-bar {
  height: 3px; border-radius: 2px; margin-bottom: .9rem;
  background: linear-gradient(90deg, var(--accent2), var(--accent3));
  transform-origin: left; animation: barGrow .6s ease both;
}
.rec-title {
  font-family: 'Bebas Neue', sans-serif; font-size: 1.25rem;
  letter-spacing: .5px; color: #fff; line-height: 1.2;
  margin-bottom: .3rem;
}
.rec-meta { font-size: .72rem; color: var(--muted); margin-bottom: .6rem; }
.rec-genres { display: flex; flex-wrap: wrap; gap: .25rem; margin-bottom: .8rem; }
.rc { padding: .12rem .45rem; border-radius: 10px; font-size: .62rem; background: var(--surface); border: 1px solid var(--border); color: var(--muted); }

/* score bars */
.sbar-row { display: flex; align-items: center; gap: .5rem; margin-top: .35rem; }
.sbar-lbl { font-size: .62rem; color: var(--muted); width: 48px; flex-shrink: 0; text-align: right; }
.sbar-track { flex: 1; height: 4px; background: var(--surface); border-radius: 2px; overflow: hidden; }
.sbar-fill { height: 100%; border-radius: 2px; transition: width .8s cubic-bezier(.4,0,.2,1); }
.sbar-val { font-size: .62rem; color: var(--muted); width: 28px; text-align: right; }

/* watchlist heart */
.wl-btn {
  position: absolute; bottom: .9rem; right: .9rem;
  background: transparent; border: 1px solid var(--border);
  border-radius: 50%; width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  font-size: .8rem; cursor: pointer; transition: all .18s;
  color: var(--muted);
}
.wl-btn.saved { border-color: var(--accent3); color: var(--accent3); background: rgba(255,101,132,.1); }

/* ── WATCHLIST PANEL ─────────────────────────────────────────── */
.wl-panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.2rem 1.4rem;
  margin-bottom: 2rem;
}
.wl-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem; letter-spacing: 1px; color: var(--accent3); margin-bottom: .8rem; }
.wl-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: .45rem 0; border-bottom: 1px solid var(--border); font-size: .82rem;
}
.wl-item:last-child { border-bottom: none; }
.wl-item-title { color: var(--text); }
.wl-item-genre { font-size: .68rem; color: var(--muted); }
.wl-remove { font-size: .7rem; color: var(--accent3); cursor: pointer; }

/* ── TABS ────────────────────────────────────────────────────── */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important; gap: .3rem; border-bottom: 1px solid var(--border) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important;
  font-family: 'Outfit', sans-serif !important; font-size: .82rem !important;
  border-radius: 0 !important; padding: .6rem 1rem !important;
  border: none !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ── CONTROLS ────────────────────────────────────────────────── */
div[data-testid="stSlider"] label { color: var(--muted) !important; font-size: .75rem !important; }
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }
div[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
div[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: .75rem !important; }
div[data-testid="stSelectbox"] > div > div {
  background: var(--card) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 10px !important;
}

.stButton > button {
  background: linear-gradient(90deg, var(--accent2), #9f97ff) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
  padding: .55rem 1.4rem !important; transition: opacity .2s !important;
  width: 100% !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── SECTION HEADER ──────────────────────────────────────────── */
.sec-hdr {
  display: flex; align-items: center; gap: .7rem;
  margin-bottom: 1.2rem;
}
.sec-hdr-line { flex: 1; height: 1px; background: var(--border); }
.sec-hdr-text {
  font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem;
  letter-spacing: 2px; color: var(--muted); white-space: nowrap;
}

/* ── EMPTY STATE ─────────────────────────────────────────────── */
.empty-state {
  text-align: center; padding: 3rem 1rem; color: var(--muted);
}
.empty-state .es-icon { font-size: 3rem; margin-bottom: .8rem; }
.empty-state .es-msg { font-size: .9rem; }

/* ── ANIMATIONS ──────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes barGrow {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
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

with st.spinner("🎬 Warming up the projection room…"):
    movies, ratings, cosine_sim, movie_idx, title_idx, collab_model = load_all()

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "user_ratings"  not in st.session_state: st.session_state.user_ratings  = {}
if "watchlist"     not in st.session_state: st.session_state.watchlist     = {}
if "active_genre"  not in st.session_state: st.session_state.active_genre  = "All"
if "active_tab"    not in st.session_state: st.session_state.active_tab    = 0

# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
wl_count = len(st.session_state.watchlist)
st.markdown(f"""
<div class="navbar">
  <div>
    <div class="nav-logo">CineMatch</div>
    <div class="nav-tagline">Hybrid · Content × Collaborative · MovieLens</div>
  </div>
  <div style="display:flex;align-items:center;gap:1rem">
    <span style="font-size:.78rem;color:var(--muted)">❤️ <b style="color:var(--accent3)">{wl_count}</b> saved</span>
    <span style="font-size:.78rem;color:var(--muted)">Mode: <b style="color:var(--accent)">{'Hybrid' if collab_model else 'Content'}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════════════
n_users = ratings["userId"].nunique()
mode_txt = "HYBRID" if collab_model else "CONTENT"
st.markdown(f"""
<div class="stats-row">
  <div class="stat-card"><div class="stat-num">{len(movies):,}</div><div class="stat-lbl">Movies</div></div>
  <div class="stat-card"><div class="stat-num">{len(ratings)//1000}K</div><div class="stat-lbl">Ratings</div></div>
  <div class="stat-card"><div class="stat-num">{n_users:,}</div><div class="stat-lbl">Users</div></div>
  <div class="stat-card"><div class="stat-num" style="color:var(--accent2)">{mode_txt}</div><div class="stat-lbl">Active Mode</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_discover, tab_rate, tab_watchlist = st.tabs(["🔍  Discover", "⭐  Rate Movies", "❤️  My Watchlist"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
with tab_discover:

    # ── Search + controls row ─────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1.2, 1])

    with ctrl1:
        search_query = st.text_input("", placeholder="🔍  Search for a movie…", label_visibility="collapsed")

    with ctrl2:
        top_n = st.slider("Results", 5, 20, 10, 1)

    with ctrl3:
        alpha = st.slider("Content weight", 0.0, 1.0, 0.5, 0.05,
                          help="1 = pure content, 0 = pure collaborative")

    # ── Genre filter chips ────────────────────────────────────────────────────
    genre_list = sorted(set(
        g for genres in movies["genres"].dropna()
        for g in genres.split("|")
        if g not in ("(no genres listed)", "IMAX")
    ))

    # Build clickable genre chips via query param trick
    all_genres = ["All"] + genre_list
    active_g = st.session_state.active_genre

    chips_html = '<div class="genre-row">'
    for g in all_genres:
        active_cls = "active" if g == active_g else ""
        chips_html += f'<span class="gchip {active_cls}" onclick="">{g}</span>'
    chips_html += '</div>'

    # Render chips as display; use selectbox for actual interactivity
    genre_sel = st.selectbox("🎭 Filter by genre", all_genres,
                              index=all_genres.index(active_g),
                              label_visibility="collapsed")
    st.session_state.active_genre = genre_sel

    # ── Movie selector ────────────────────────────────────────────────────────
    filtered_movies = movies if genre_sel == "All" else movies[movies["genres"].str.contains(genre_sel, na=False)]

    if search_query:
        mask = filtered_movies["clean_title"].str.contains(search_query, case=False, na=False)
        filtered_movies = filtered_movies[mask]

    movie_titles = sorted(filtered_movies["clean_title"].dropna().unique().tolist())

    if not movie_titles:
        st.markdown('<div class="empty-state"><div class="es-icon">🎬</div><div class="es-msg">No movies match your search. Try a different query.</div></div>', unsafe_allow_html=True)
        st.stop()

    default_idx = movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0
    selected_title = st.selectbox("🎬 Select seed movie", movie_titles, index=default_idx)

    seed_row = movies[movies["clean_title"] == selected_title].iloc[0]
    seed_id  = int(seed_row["movieId"])

    # ── Seed banner ───────────────────────────────────────────────────────────
    year_d  = str(int(seed_row["year"])) if pd.notna(seed_row.get("year")) else "N/A"
    genres_d = seed_row["genres"].replace("|", " · ") if pd.notna(seed_row["genres"]) else ""
    genre_chips_seed = "".join(
        f'<span class="seed-genre-chip">{g.strip()}</span>'
        for g in str(seed_row["genres"]).split("|")
        if g.strip() not in ("(no genres listed)", "IMAX", "nan")
    )
    stars = "★" * round(seed_row["avg_rating"] / 1) + "☆" * max(0, 5 - round(seed_row["avg_rating"]))

    st.markdown(f"""
<div class="seed-banner">
  <div class="seed-poster">🎞</div>
  <div class="seed-info">
    <div class="seed-title">{seed_row['clean_title']}</div>
    <div class="seed-sub">📅 {year_d} &nbsp;·&nbsp; MovieLens ID #{seed_id}</div>
    <div class="seed-genres">{genre_chips_seed}</div>
  </div>
  <div class="seed-stats">
    <div class="seed-rating">{seed_row['avg_rating']:.1f}</div>
    <div style="color:var(--accent);font-size:.8rem">★★★★☆</div>
    <div class="seed-rcount">{int(seed_row['rating_count']):,} ratings</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Blend indicator ───────────────────────────────────────────────────────
    dot_pct = int(alpha * 100)
    st.markdown(f"""
<div class="blend-display">
  <div>
    <div class="blend-label">Collaborative</div>
    <div style="font-size:.7rem;color:var(--accent2)">{(1-alpha):.0%}</div>
  </div>
  <div class="blend-bar-track">
    <div class="blend-dot" style="left:calc({dot_pct}% - 7px)"></div>
  </div>
  <div>
    <div class="blend-label">Content-Based</div>
    <div style="font-size:.7rem;color:var(--accent3)">{alpha:.0%}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Compute recommendations ───────────────────────────────────────────────
    with st.spinner("🔮 Finding your next favourite films…"):
        recs = hybrid_recommend(
            seed_movie_id=seed_id,
            user_ratings=st.session_state.user_ratings,
            movies=movies,
            cosine_sim=cosine_sim,
            movie_idx=movie_idx,
            collab_model=collab_model if SURPRISE_AVAILABLE else None,
            top_n=top_n,
            alpha=alpha,
        )

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="sec-hdr">
  <div class="sec-hdr-line"></div>
  <div class="sec-hdr-text">TOP {top_n} PICKS FOR "{selected_title.upper()}"</div>
  <div class="sec-hdr-line"></div>
</div>
""", unsafe_allow_html=True)

    # ── Recommendation cards ──────────────────────────────────────────────────
    if recs.empty:
        st.markdown('<div class="empty-state"><div class="es-icon">😔</div><div class="es-msg">No recommendations found. Try a different movie.</div></div>', unsafe_allow_html=True)
    else:
        # Render cards in a 3-column grid
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            mid = int(row["movieId"])
            in_wl = mid in st.session_state.watchlist
            wl_cls = "saved" if in_wl else ""
            wl_icon = "❤️" if in_wl else "🤍"

            yr = str(int(row["year"])) if pd.notna(row.get("year")) else "—"
            avg_r = row.get("avg_rating", 0)
            c_s   = row.get("content_score", 0)
            cf_s  = row.get("collab_score", 0)
            h_s   = row.get("hybrid_score", 0)
            rc    = int(row.get("rating_count", 0))

            genre_chips = "".join(
                f'<span class="rc">{g.strip()}</span>'
                for g in str(row.get("genres","")).split("|")
                if g.strip() not in ("(no genres listed)", "IMAX", "nan", "")
            )

            def bar(val, color, delay):
                pct = int(val * 100)
                return (f'<div class="sbar-track"><div class="sbar-fill" '
                        f'style="width:{pct}%;background:{color};'
                        f'animation-delay:{delay}s"></div></div>')

            delay = i * 0.05

            with cols[i % 3]:
                st.markdown(f"""
<div class="rec-card" style="animation-delay:{delay}s">
  <div class="rec-accent-bar" style="animation-delay:{delay+.1}s"></div>
  <span class="rec-rank">#{i+1}</span>
  <div class="rec-title">{row['clean_title']}</div>
  <div class="rec-meta">📅 {yr} &nbsp;·&nbsp; ⭐ {avg_r:.1f} ({rc:,})</div>
  <div class="rec-genres">{genre_chips}</div>
  <div class="sbar-row">
    <span class="sbar-lbl">Content</span>
    {bar(c_s, '#6c63ff', delay+.2)}
    <span class="sbar-val">{c_s:.2f}</span>
  </div>
  <div class="sbar-row">
    <span class="sbar-lbl">Collab</span>
    {bar(cf_s, '#ff6584', delay+.25)}
    <span class="sbar-val">{cf_s:.2f}</span>
  </div>
  <div class="sbar-row">
    <span class="sbar-lbl">Hybrid</span>
    {bar(h_s, '#e8b14a', delay+.3)}
    <span class="sbar-val">{h_s:.2f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                # Watchlist toggle button per card
                btn_label = f"{'❤️ Saved' if in_wl else '🤍 Save'}  —  {row['clean_title'][:22]}"
                if st.button(btn_label, key=f"wl_{mid}"):
                    if in_wl:
                        del st.session_state.watchlist[mid]
                    else:
                        st.session_state.watchlist[mid] = {
                            "title": row["clean_title"],
                            "genres": row.get("genres", ""),
                            "year": yr,
                            "rating": avg_r,
                        }
                    st.rerun()

        # ── Score chart ───────────────────────────────────────────────────────
        st.markdown("""
<div class="sec-hdr" style="margin-top:2rem">
  <div class="sec-hdr-line"></div>
  <div class="sec-hdr-text">SCORE BREAKDOWN</div>
  <div class="sec-hdr-line"></div>
</div>
""", unsafe_allow_html=True)
        chart_df = recs[["clean_title","content_score","collab_score","hybrid_score"]].set_index("clean_title")
        st.bar_chart(chart_df, height=280, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RATE MOVIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_rate:
    st.markdown("""
<div class="sec-hdr">
  <div class="sec-hdr-line"></div>
  <div class="sec-hdr-text">RATE MOVIES TO PERSONALISE COLLABORATIVE FILTERING</div>
  <div class="sec-hdr-line"></div>
</div>
""", unsafe_allow_html=True)

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        genre_filter_rate = st.selectbox("Filter by genre", ["All"] + genre_list, key="rate_genre")
        rate_pool = movies if genre_filter_rate == "All" else movies[movies["genres"].str.contains(genre_filter_rate, na=False)]
        rate_search = st.text_input("", placeholder="🔍  Search movie to rate…", key="rate_search", label_visibility="collapsed")
        if rate_search:
            rate_pool = rate_pool[rate_pool["clean_title"].str.contains(rate_search, case=False, na=False)]
        rate_titles = sorted(rate_pool["clean_title"].dropna().unique().tolist())
        rate_movie = st.selectbox("Choose movie", rate_titles, label_visibility="collapsed")

    with rc2:
        rating_val = st.slider("Your Rating ⭐", 0.5, 5.0, 3.5, 0.5)
        stars_display = "⭐" * int(rating_val) + ("½" if rating_val % 1 else "")
        st.markdown(f'<div style="text-align:center;font-size:1.4rem;margin:.3rem 0">{stars_display}</div>', unsafe_allow_html=True)
        if st.button("➕  Add Rating"):
            row = movies[movies["clean_title"] == rate_movie]
            if not row.empty:
                mid = int(row.iloc[0]["movieId"])
                st.session_state.user_ratings[mid] = rating_val
                st.success(f"✅  Rated **{rate_movie}** → {rating_val} ⭐")

    # Display rated movies
    if st.session_state.user_ratings:
        st.markdown(f"""
<div class="sec-hdr" style="margin-top:1.5rem">
  <div class="sec-hdr-line"></div>
  <div class="sec-hdr-text">YOUR {len(st.session_state.user_ratings)} RATINGS</div>
  <div class="sec-hdr-line"></div>
</div>
""", unsafe_allow_html=True)

        rated_ids = list(st.session_state.user_ratings.keys())
        rated_df = movies[movies["movieId"].isin(rated_ids)][["movieId","clean_title","genres","avg_rating"]].copy()
        rated_df["your_rating"] = rated_df["movieId"].map(st.session_state.user_ratings)

        r_cols = st.columns(3)
        for i, (_, rrow) in enumerate(rated_df.iterrows()):
            with r_cols[i % 3]:
                yr_rating = rrow["your_rating"]
                star_str = "⭐" * int(yr_rating)
                st.markdown(f"""
<div class="rec-card" style="padding:.9rem 1rem">
  <div class="rec-title" style="font-size:1rem">{rrow['clean_title']}</div>
  <div class="rec-meta">{rrow['genres'].replace('|',' · ')[:50]}</div>
  <div style="margin-top:.4rem;font-size:1rem">{star_str} <span style="color:var(--accent);font-size:.9rem">{yr_rating}</span></div>
</div>
""", unsafe_allow_html=True)

        if st.button("🗑️  Clear All Ratings", key="clear_ratings"):
            st.session_state.user_ratings = {}
            st.rerun()
    else:
        st.markdown('<div class="empty-state"><div class="es-icon">🎭</div><div class="es-msg">Rate some movies above to unlock personalised collaborative recommendations.</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
with tab_watchlist:
    st.markdown("""
<div class="sec-hdr">
  <div class="sec-hdr-line"></div>
  <div class="sec-hdr-text">MY WATCHLIST</div>
  <div class="sec-hdr-line"></div>
</div>
""", unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.markdown('<div class="empty-state"><div class="es-icon">❤️</div><div class="es-msg">Your watchlist is empty.<br>Hit the Save button on any recommendation card to add movies here.</div></div>', unsafe_allow_html=True)
    else:
        wl_cols = st.columns(3)
        to_remove = []
        for i, (mid, info) in enumerate(st.session_state.watchlist.items()):
            with wl_cols[i % 3]:
                genre_chips_wl = "".join(
                    f'<span class="rc">{g.strip()}</span>'
                    for g in str(info.get("genres","")).split("|")
                    if g.strip() not in ("(no genres listed)", "IMAX", "nan", "")
                )
                st.markdown(f"""
<div class="rec-card">
  <div class="rec-accent-bar" style="background:linear-gradient(90deg,var(--accent3),var(--accent))"></div>
  <div class="rec-title">{info['title']}</div>
  <div class="rec-meta">📅 {info['year']} &nbsp;·&nbsp; ⭐ {info['rating']:.1f}</div>
  <div class="rec-genres">{genre_chips_wl}</div>
</div>
""", unsafe_allow_html=True)
                if st.button(f"🗑️ Remove", key=f"rm_{mid}"):
                    to_remove.append(mid)

        for mid in to_remove:
            del st.session_state.watchlist[mid]
        if to_remove:
            st.rerun()

        st.markdown("---")
        if st.button("🗑️  Clear Entire Watchlist"):
            st.session_state.watchlist = {}
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2rem 0 .5rem;color:var(--muted);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase">
  CineMatch v2 &nbsp;·&nbsp; MovieLens Small Dataset &nbsp;·&nbsp; scikit-learn + Streamlit
</div>
""", unsafe_allow_html=True)
