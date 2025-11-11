from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Local imports
from recommender_utils import (
    get_hybrid_recommendations,
    get_collab_recommendations,
    get_content_recommendations_by_content_id,
    load_items
)

# === Flask Setup ===
app = Flask(__name__)

# === Model Loader ===
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

def load_models():
    models = {}

    # TF-IDF and cosine similarity
    try:
        models['tfidf'] = joblib.load(MODELS_DIR / 'tfidf_vectorizer.joblib')
    except:
        models['tfidf'] = None
    try:
        models['cosine_sim'] = np.load(MODELS_DIR / 'cosine_sim.npy')
    except:
        models['cosine_sim'] = None
    try:
        with open(MODELS_DIR / 'items_map.pkl', 'rb') as f:
            mm = pickle.load(f)
            models['content_id_to_idx'] = mm.get('content_id_to_idx', {})
            models['idx_to_content_id'] = mm.get('idx_to_content_id', {})
    except:
        models['content_id_to_idx'] = {}
        models['idx_to_content_id'] = {}

    # CF SVD
    try:
        with open(MODELS_DIR / 'cf_svd.pkl', 'rb') as f:
            cf = pickle.load(f)
            models.update(cf)
    except:
        models.update({
            'user_to_idx': {},
            'item_to_idx': {},
            'user_factors': None,
            'item_factors': None
        })

    # Items metadata
    try:
        models['items_df'] = pd.read_csv(MODELS_DIR / 'items_clean.csv')
    except:
        try:
            models['items_df'] = pd.read_csv(APP_DIR / 'data' / 'items.csv')
        except:
            models['items_df'] = pd.DataFrame()

    # Interactions
    try:
        models['interactions'] = pd.read_csv(MODELS_DIR / 'interactions_clean.csv')
    except:
        try:
            models['interactions'] = pd.read_csv(APP_DIR / 'data' / 'interactions.csv')
        except:
            models['interactions'] = pd.DataFrame()

    # Safely convert numeric columns
    for df_key in ['items_df', 'interactions']:
        df = models.get(df_key)
        if not df.empty and 'content_id' in df.columns:
            df['content_id'] = pd.to_numeric(df['content_id'], errors='coerce')
            df.dropna(subset=['content_id'], inplace=True)
            df['content_id'] = df['content_id'].astype(int)
    return models


MODELS = load_models()

# === Helper ===
def get_item_meta(content_ids):
    df = MODELS.get('items_df')
    out = []
    for cid in content_ids:
        row = df[df['content_id'] == int(cid)]
        if not row.empty:
            r = row.iloc[0].to_dict()
            out.append({
                'content_id': int(r['content_id']),
                'title': r.get('title', ''),
                'description': r.get('description', '')
            })
        else:
            out.append({'content_id': int(cid), 'title': '', 'description': ''})
    return out


# === Frontend Routes ===
@app.route("/")
def index():
    items = load_items()
    return render_template("index.html", items=items.to_dict(orient="records"))

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.form
    user_id = int(data.get("user_id"))
    method = data.get("method", "hybrid")
    top_n = int(data.get("top_n", 10))

    if method == "hybrid":
        recs = get_hybrid_recommendations(user_id, top_n=top_n)
    elif method == "collab":
        recs = get_collab_recommendations(user_id, top_n=top_n)
    elif method == "content":
        content_id = int(data.get("content_id"))
        recs = get_content_recommendations_by_content_id(content_id, top_n=top_n)
    else:
        return "Unknown method", 400

    return render_template(
        "index.html",
        items=load_items().to_dict(orient="records"),
        recs=recs.to_dict(orient="records")
    )

# === API Routes ===
@app.route("/recommendations/content/<int:content_id>")
def content_recommendations(content_id):
    top_n = int(request.args.get('top_n', 10))
    cid_to_idx = MODELS.get('content_id_to_idx', {})
    idx_to_cid = MODELS.get('idx_to_content_id', {})
    cosine_sim = MODELS.get('cosine_sim')

    if cosine_sim is None or not cid_to_idx:
        return jsonify({'error': 'Content model unavailable'}), 500

    idx = cid_to_idx.get(int(content_id))
    if idx is None:
        return jsonify({'error': f'content_id {content_id} not found'}), 404

    sims = cosine_sim[idx]
    order = np.argsort(-sims)
    order = order[order != idx]
    top_idxs = order[:top_n]
    top_content_ids = [int(idx_to_cid.get(int(i))) for i in top_idxs]
    items = get_item_meta(top_content_ids)
    return jsonify({'content_id': int(content_id), 'recommendations': items})


@app.route("/recommendations/cf/<int:user_id>")
def cf_recommendations(user_id):
    top_n = int(request.args.get('top_n', 10))
    user_to_idx = MODELS.get('user_to_idx', {})
    item_to_idx = MODELS.get('item_to_idx', {})
    user_factors = MODELS.get('user_factors')
    item_factors = MODELS.get('item_factors')
    interactions = MODELS.get('interactions')

    if user_factors is None or item_factors is None:
        return jsonify({'error': 'CF model unavailable'}), 500

    if int(user_id) not in user_to_idx:
        if interactions is None or interactions.empty:
            return jsonify({'recommendations': []})
        popular = interactions['content_id'].value_counts().head(top_n).index.tolist()
        items = get_item_meta(popular)
        return jsonify({'user_id': int(user_id), 'recommendations': items})

    uidx = user_to_idx[int(user_id)]
    scores = np.dot(user_factors[uidx], item_factors.T)

    seen = set()
    if interactions is not None and not interactions.empty:
        seen_rows = interactions[interactions['user_id'] == int(user_id)]
        seen = set(seen_rows['content_id'].astype(int).tolist())

    idx_to_content = {v: k for k, v in item_to_idx.items()}
    scored = [(cid, float(scores[i])) for i, cid in idx_to_content.items() if cid not in seen]
    scored.sort(key=lambda x: -x[1])
    top = [int(cid) for cid, _ in scored[:top_n]]
    items = get_item_meta(top)
    return jsonify({'user_id': int(user_id), 'recommendations': items})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
