# train_model.py
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

USERS_CSV = DATA_DIR / "users.csv"
ITEMS_CSV = DATA_DIR / "items.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

# 1) Load
users = pd.read_csv(USERS_CSV)
items = pd.read_csv(ITEMS_CSV)
interactions = pd.read_csv(INTERACTIONS_CSV)

# 2) Basic cleaning
# drop exact duplicate interaction rows
interactions = interactions.drop_duplicates()

# fill missing viewed_time with median
if 'viewed_time(min)' in interactions.columns:
    interactions['viewed_time(min)'] = interactions['viewed_time(min)'].fillna(interactions['viewed_time(min)'].median())

# clip ratings to valid range 1..5 (handles outliers)
if 'rating' in interactions.columns:
    interactions['rating'] = pd.to_numeric(interactions['rating'], errors='coerce')
    interactions['rating'] = interactions['rating'].fillna(0)
    interactions.loc[interactions['rating'] > 5, 'rating'] = 5
    interactions.loc[interactions['rating'] < 1, 'rating'] = 1

# 3) Prepare item text for TF-IDF
# make sure columns exist
items['title'] = items['title'].fillna('').astype(str)
items['description'] = items['description'].fillna('').astype(str)
items['sdg_goal'] = items['sdg_goal'].fillna('').astype(str)
items['category'] = items['category'].fillna('').astype(str)

items['combined_text'] = items['title'] + " " + items['description'] + " " + items['sdg_goal'] + " " + items['category']

# 4) TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(items['combined_text'])

# 5) Cosine similarity (item x item)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 6) Map content_id -> item index
items = items.reset_index(drop=True)
content_id_to_idx = {int(cid): idx for idx, cid in enumerate(items['content_id'].astype(int).tolist())}
idx_to_content_id = {v: k for k, v in content_id_to_idx.items()}

# Save vectorizer, cosine_sim, items mapping
dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
np.save(MODELS_DIR / "cosine_sim.npy", cosine_sim)
with open(MODELS_DIR / "items_map.pkl", "wb") as f:
    pickle.dump({'content_id_to_idx': content_id_to_idx, 'idx_to_content_id': idx_to_content_id}, f)

print("Saved TF-IDF vectorizer, cosine_sim, and mappings to models/")

# 7) Collaborative filtering (TruncatedSVD-based approximation)
# We'll build a user-item rating matrix and apply TruncatedSVD to obtain
# low-rank user and item factors. This avoids the `surprise` dependency.
rating_df = interactions[['user_id', 'content_id', 'rating']].copy()
rating_df['rating'] = pd.to_numeric(rating_df['rating'], errors='coerce').fillna(0)

# Map users/items to indices
user_ids = rating_df['user_id'].unique().tolist()
item_ids = items['content_id'].astype(int).unique().tolist()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {int(it): i for i, it in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)

from scipy.sparse import lil_matrix, csr_matrix

# Build full interaction sparse matrix (users x items)
full_mat = lil_matrix((n_users, n_items), dtype=float)
for _, row in rating_df.iterrows():
    u = user_to_idx.get(row['user_id'])
    it = item_to_idx.get(int(row['content_id']))
    if u is None or it is None:
        continue
    full_mat[u, it] = row['rating']
full_mat = csr_matrix(full_mat)

# Train/test split on interactions (rows)
train_rows, test_rows = train_test_split(rating_df, test_size=0.2, random_state=42)

# Build train matrix
train_mat = lil_matrix((n_users, n_items), dtype=float)
for _, row in train_rows.iterrows():
    u = user_to_idx.get(row['user_id'])
    it = item_to_idx.get(int(row['content_id']))
    if u is None or it is None:
        continue
    train_mat[u, it] = row['rating']
train_mat = csr_matrix(train_mat)

# Fit TruncatedSVD on the train matrix
n_components = min(50, min(n_users - 1 if n_users > 1 else 1, n_items - 1 if n_items > 1 else 1))
if n_components < 1:
    n_components = 1
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(train_mat)  # shape (n_users, n_components)
item_factors = svd.components_.T  # shape (n_items, n_components)

# Evaluate on test interactions
preds = []
truths = []
for _, row in test_rows.iterrows():
    u = user_to_idx.get(row['user_id'])
    it = item_to_idx.get(int(row['content_id']))
    if u is None or it is None:
        continue
    pred = float(user_factors[u].dot(item_factors[it]))
    preds.append(pred)
    truths.append(float(row['rating']))

if len(preds) > 0:
    rmse = np.sqrt(mean_squared_error(truths, preds))
    print(f"TruncatedSVD CF RMSE on testset: {rmse:.4f}")
else:
    print("No test interactions available to evaluate CF model.")

# Save CF model (factors and mappings)
with open(MODELS_DIR / "cf_svd.pkl", "wb") as f:
    pickle.dump({
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'svd_model': svd,
    }, f)

print("Saved CF TruncatedSVD model to models/cf_svd.pkl")

# 8) Export cleaned datasets to models folder for easy loading by app (optional)
items.to_csv(MODELS_DIR / "items_clean.csv", index=False)
users.to_csv(MODELS_DIR / "users_clean.csv", index=False)
interactions.to_csv(MODELS_DIR / "interactions_clean.csv", index=False)

print("Training complete. Models and cleaned CSVs saved in models/")
