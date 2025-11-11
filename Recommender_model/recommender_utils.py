# recommender_utils.py
import pickle
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = "models"

# Load resources (lazy-load helpers)
def load_items():
    return pd.read_csv(f"{MODELS_DIR}/items_clean.csv")

def load_svd():
    """Load CF artifacts saved by the TruncatedSVD-based training.

    Returns a dict with keys: user_to_idx, item_to_idx, user_factors, item_factors, svd_model
    """
    with open(f"{MODELS_DIR}/cf_svd.pkl", "rb") as f:
        return pickle.load(f)

def load_tfidf_and_sim():
    tfidf = load(f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    cosine_sim = np.load(f"{MODELS_DIR}/cosine_sim.npy")
    with open(f"{MODELS_DIR}/items_map.pkl", "rb") as f:
        maps = pickle.load(f)
    return tfidf, cosine_sim, maps

# Content-based recommendations given an item content_id or title
def get_content_recommendations_by_content_id(content_id, top_n=5):
    items = load_items()
    tfidf, cosine_sim, maps = load_tfidf_and_sim()
    content_id_to_idx = maps['content_id_to_idx']
    if content_id not in content_id_to_idx:
        raise ValueError("content_id not found")
    idx = content_id_to_idx[int(content_id)]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n+1]
    rec_indices = [i for i, _ in sim_scores]
    return items.iloc[rec_indices][['content_id', 'title', 'sdg_goal', 'category']].copy()

# Collaborative top-n predictions for user (predict for all items user hasn't rated)
def get_collab_recommendations(user_id, top_n=10):
    items = load_items()
    cf = load_svd()
    user_to_idx = cf.get('user_to_idx', {})
    item_to_idx = cf.get('item_to_idx', {})
    user_factors = cf.get('user_factors')
    item_factors = cf.get('item_factors')

    preds = []
    # If we have factors and the user is known, score by dot-product
    if user_factors is not None and item_factors is not None and int(user_id) in user_to_idx:
        uidx = user_to_idx[int(user_id)]
        for _, row in items.iterrows():
            content_id = int(row['content_id'])
            item_idx = item_to_idx.get(content_id)
            if item_idx is None:
                est = 0.0
            else:
                est = float(user_factors[uidx].dot(item_factors[item_idx]))
            preds.append((content_id, row.get('title', ''), est))
    else:
        # fallback: recommend most popular items (by interaction count) if CF not available or unknown user
        try:
            interactions = pd.read_csv(f"{MODELS_DIR}/interactions_clean.csv")
            popular = interactions['content_id'].value_counts().index.tolist()
        except Exception:
            popular = items['content_id'].astype(int).tolist()
        for cid in popular:
            row = items[items['content_id'].astype(int) == int(cid)]
            title = row.iloc[0]['title'] if not row.empty else ''
            preds.append((int(cid), title, 0.0))
    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(preds_sorted[:top_n], columns=['content_id', 'title', 'pred_rating'])
    return df

# Hybrid: weighted combination of content score + collab predicted rating
def get_hybrid_recommendations(user_id, top_n=10, w_content=0.6, w_collab=0.4):
    items = load_items()
    cf = load_svd()
    _, cosine_sim, maps = load_tfidf_and_sim()
    content_id_to_idx = maps['content_id_to_idx']
    user_to_idx = cf.get('user_to_idx', {})
    item_to_idx = cf.get('item_to_idx', {})
    user_factors = cf.get('user_factors')
    item_factors = cf.get('item_factors')

    hybrid_scores = []
    for idx in range(len(items)):
        content_id = int(items.iloc[idx]['content_id'])
        # content score = average similarity to all items (simple measure)
        content_score = float(cosine_sim[idx].mean()) if cosine_sim is not None else 0.0
        # collaborative score: dot product if factors exist and user known, else 0
        collab_score = 0.0
        if user_factors is not None and item_factors is not None and int(user_id) in user_to_idx:
            uidx = user_to_idx[int(user_id)]
            item_idx = item_to_idx.get(content_id)
            if item_idx is not None:
                collab_score = float(user_factors[uidx].dot(item_factors[item_idx]))
        # normalize collab_score conservatively to 0..1 by min-max or rating scale assumptions
        # Here we assume original ratings were roughly in 1..5; if the factors produce similar scale, clip and normalize
        collab_score_norm = (collab_score - 1.0) / 4.0 if collab_score is not None else 0.0
        hybrid_score = w_content * content_score + w_collab * collab_score_norm
        hybrid_scores.append((content_id, items.iloc[idx]['title'], hybrid_score, content_score, collab_score))

    hybrid_sorted = sorted(hybrid_scores, key=lambda x: x[2], reverse=True)[:top_n]
    df = pd.DataFrame(hybrid_sorted, columns=['content_id', 'title', 'hybrid_score', 'content_score', 'collab_score'])
    return df
