# 🎬 CineMatch — Hybrid Movie Recommender

A production-ready Streamlit app combining **Content-Based** and **Collaborative Filtering** to recommend movies from the [MovieLens Small dataset](https://grouplens.org/datasets/movielens/latest/).
<img width="1880" height="907" alt="mvc4" src="https://github.com/user-attachments/assets/249f72d8-d024-4c67-9303-51fdc8ededfc" />
<img width="1883" height="917" alt="mvc3" src="https://github.com/user-attachments/assets/84fdb4ea-db40-4ec3-ac00-4264fc35bc09" />
<img width="1901" height="900" alt="mvc2" src="https://github.com/user-attachments/assets/5574220b-36c9-4235-9973-6e52838e9aa6" />
<img width="1901" height="907" alt="mvc1" src="https://github.com/user-attachments/assets/cd77bd6f-72f6-4d54-816e-103df89cff68" />

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

## The Code 
<img width="1472" height="902" alt="mvc code4" src="https://github.com/user-attachments/assets/4302371f-9a57-4d8f-b7ba-68da287528e1" />
<img width="1463" height="892" alt="mvc code3" src="https://github.com/user-attachments/assets/f33fe4e6-e9fd-43dd-8e43-1d839bef7080" />
<img width="1451" height="897" alt="mvc code1" src="https://github.com/user-attachments/assets/01c15aa6-f7cb-42ae-8fb1-b30d5b3ca8e5" />
<img width="1481" height="946" alt="mvc code " src="https://github.com/user-attachments/assets/8cc1f92b-f77b-48a3-ae20-62e834d902c8" />


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
