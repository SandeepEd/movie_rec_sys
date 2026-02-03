# Movie Recommendation System

A progressive implementation of movie recommendation systems using the MovieLens 100k dataset, built with PyTorch.

## Project Overview

This project implements 5 progressively complex recommendation approaches:

1. **Genre-Based Baseline** - Simple popularity-based recommendations within user's preferred genres
2. **Content-Based Filtering** - Recommendations based on movie features (genres, title, release year)
3. **Collaborative Filtering** - Matrix factorization and neural collaborative filtering
4. **Two-Tower Architecture** - Deep learning approach with separate user and item embedding towers
5. **Ranking & Reranking** - Two-stage pipeline combining retrieval and ranking

## Project Structure

```
recommendation_system/
├── data/
│   └── ml-100k/           # MovieLens 100k dataset
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_genre_baseline.ipynb         # Phase 1: Genre-based recommendations
│   ├── 03_content_based.ipynb          # Phase 2: Content-based filtering
│   ├── 04_collaborative.ipynb          # Phase 3: Collaborative filtering
│   ├── 05_two_tower.ipynb              # Phase 4: Two-tower architecture
│   └── 06_ranking.ipynb                # Phase 5: Ranking model
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessing.py   # Feature engineering
│   ├── models/
│   │   ├── genre_baseline.py  # Genre-based recommender
│   │   ├── content_based.py   # Content-based recommender
│   │   ├── collaborative.py   # Collaborative filtering models
│   │   ├── two_tower.py       # Two-tower neural network
│   │   └── ranker.py          # Ranking model
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       └── helpers.py         # Helper functions
├── models/
│   └── saved/                 # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The MovieLens 100k dataset is already in `data/ml-100k/`

3. Start with the notebooks in order:
   - Begin with `01_eda.ipynb` for data exploration
   - Progress through each phase sequentially

## Dataset

**MovieLens 100k Dataset**
- 100,000 ratings from 943 users on 1,682 movies
- Ratings scale: 1-5
- User demographics: age, gender, occupation
- Movie metadata: title, release date, genres
- Pre-split train/test sets provided

## Implementation Phases

### Phase 1: Baseline Popularity-Based Recommender
Simple recommender using global movie popularity.

### Phase 2: Content-Based Filtering
Uses movie features (genres, title TF-IDF, release year) to find similar items.

### Phase 3: Collaborative Filtering
Implements matrix factorization using PyTorch to learn user and item embeddings.

### Phase 4: Two-Tower Architecture
Deep learning model with separate neural networks for users and items, enabling efficient retrieval.

### Phase 5: Ranking & Reranking
Two-stage pipeline: fast retrieval using two-tower, then precise ranking with rich features.

## Evaluation Metrics

All models will be evaluated using:
- Precision@K and Recall@K
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)
- Coverage and diversity metrics