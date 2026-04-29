# 🎬 CineMatch — Hybrid Movie Recommender

A production-ready Streamlit app combining **Content-Based** and **Collaborative Filtering** to recommend movies from the [MovieLens Small dataset](https://grouplens.org/datasets/movielens/latest/).

---

## 📁 Project Structure

```
movie-recommender/
├── app.py                  # Main Streamlit UI
├── data_loader.py          # Downloads & preprocesses MovieLens data
├── content_based.py        # TF-IDF + Cosine Similarity (content filtering)
├── collaborative.py        # SVD Matrix Factorization (collaborative filtering)
├── hybrid.py               # Weighted hybrid score combiner
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server settings
└── README.md
```

---

## 🚀 Deploy on Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: CineMatch hybrid recommender"
git remote add origin https://github.com/YOUR_USERNAME/movie-recommender.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy!**

> The app automatically downloads the MovieLens dataset on first run (~6 MB).

---

## 🧠 How It Works

### Content-Based Filtering
- Combines movie **genres** and user **tags** into a text feature
- Builds a **TF-IDF matrix** and computes **cosine similarity** between all movies
- Returns movies most similar to your selected seed movie

### Collaborative Filtering
- Uses **SVD (Singular Value Decomposition)** via `scikit-surprise`
- Trained on 100k+ real user ratings from the MovieLens dataset
- Predicts ratings a user would give to unseen movies

### Hybrid Blending
- Normalises both scores to [0, 1] using **MinMaxScaler**
- Combines them: `hybrid_score = α × content + (1−α) × collab`
- The **α slider** in the sidebar lets you control the blend live

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` / `numpy` | Data processing |
| `scikit-learn` | TF-IDF, cosine similarity, normalisation |
| `scikit-surprise` | SVD collaborative filtering |
| `requests` | Dataset download |

---

## 🎛️ Features

- 🔍 **Search any of 9,000+ movies** as a seed
- ⭐ **Rate movies you've seen** to personalise collaborative recommendations
- ⚖️ **Live alpha slider** to blend content vs collaborative weight
- 📊 **Score breakdown bars** (content / collab / hybrid) per recommendation
- 🌓 Custom dark cinema-themed UI
