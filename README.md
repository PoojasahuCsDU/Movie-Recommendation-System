# 🎬 Movie Recommendation System — Hybrid (Content + Collaborative)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)]()

## 📘 Overview
This project builds a **Hybrid Movie Recommendation System** that combines **Content-Based Filtering** and **Collaborative Filtering** to provide personalized movie suggestions.

It demonstrates end-to-end machine-learning workflow:  
data preprocessing → feature extraction → model building → evaluation → hybrid fusion.

---

## 🚀 Key Features
- **Content-Based Model:** Uses TF-IDF + Cosine Similarity on movie metadata (overview, cast, genres, keywords).  
- **Collaborative Filtering:** Uses **Singular Value Decomposition (SVD)** via the Surprise library to learn user–item interactions.  
- **Hybrid Approach:** Merges both models to recommend movies most relevant to each user.  
- **Evaluation Metrics:** RMSE (< 1), Precision@K, Recall@K.  
- **Dataset:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/) & TMDB metadata (≈ 45 000 movies).

---

## 📊 Project Workflow
1. **Data Preprocessing:** Cleaning, handling missing values, merging metadata & ratings.  
2. **Feature Engineering:** TF-IDF Vectorization → Cosine Similarity; Vote Weighting using IMDB formula.  
3. **Collaborative Model:** Matrix Factorization (SVD) on user–movie ratings.  
4. **Hybrid System:** Weighted combination of content & collaborative scores.  
5. **Evaluation:** RMSE, Precision@10, NDCG@10.  
6. **Visualization:** Distribution of ratings, similarity heatmaps, top recommendations.

---

## 📂 Dataset Description
| File | Description |
|------|--------------|
| `movies_metadata.csv` | 45 000 movies — titles, genres, plots, release dates, ratings |
| `keywords.csv` | Plot keywords (JSON format) |
| `credits.csv` | Cast and Crew information |
| `ratings_small.csv` | 100 000 ratings from 700 users |

---

## 🧠 Techniques Used
- **TF-IDF Vectorization**  
- **Cosine Similarity**  
- **SVD (Singular Value Decomposition)**  
- **Hybrid Weighted Scoring**  
- **Python Libraries:** pandas, numpy, scikit-learn, surprise, matplotlib, seaborn  

---

## 📈 Results
| Model | RMSE | Precision@10 | NDCG@10 |
|-------|------:|--------------:|--------:|
| Popularity Baseline | 1.23 | 0.05 | 0.07 |
| Content-Based | 0.98 | 0.12 | 0.18 |
| SVD (MF) | **0.91** | 0.16 | 0.24 |
| Hybrid Model | **0.89** | **0.19** | **0.27** |

> The hybrid model produced the most personalized and accurate recommendations.

---

## 🧾 How to Run
```bash
# clone repository
git clone https://github.com/PoojasahuCsDU/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# create environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# open Jupyter notebook
jupyter notebook Movie_recommendation_System-Project.ipynb
