# Hybrid SDG Recommender

A small hybrid recommender system focused on Sustainable Development Goals (SDGs). This project demonstrates a lightweight hybrid approach combining content-based (TF-IDF + cosine similarity) and collaborative filtering (TruncatedSVD) methods to recommend SDG-related content to users.

## Highlights
- Content-based recommendations using TF-IDF on item text (title, description, SDG, category).
- Collaborative filtering using truncated SVD on a user-item rating matrix.
- Hybrid scoring that blends content similarity and collaborative predictions.
- Simple Flask app to get recommendations via web UI and REST endpoints.
- Minimal dependencies and easy-to-run training script.

## Repository structure
- Recommender_model/
  - app.py — Flask application serving the web UI and API endpoints.
  - recommender_utils.py — helper functions for content/collab/hybrid recommenders.
  - train_model.py — training script to create TF-IDF, cosine similarity and CF artifacts.
  - models/ — generated model artifacts and cleaned CSVs (tfidf_vectorizer.joblib, cosine_sim.npy, cf_svd.pkl, items_clean.csv, interactions_clean.csv, ...)
  - data/ — raw CSV data (items.csv, users.csv, interactions.csv).
  - templates/ — UI templates (index.html) and static assets.

## Quick start

1. Clone the repository
   git clone https://github.com/aimlkhhushi1521/Recommender_system.git
   cd Recommender_model

2. Create a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows

3. Install dependencies
   pip install -r requirements.txt

4. Train models and prepare artifacts (this saves things to `models/`)
   python train_model.py

5. Run the Flask app
   python app.py
   - The app runs by default at http://127.0.0.1:5000

## Web UI
Open http://127.0.0.1:5000 and use the form to:
- Select a user ID and method (hybrid, collaborative, or content-based) and get top-N recommendations.

## API Endpoints
- GET /recommendations/content/<content_id>?top_n=10
  - Returns top-N content-based recommendations for the given content_id.
- GET /recommendations/cf/<user_id>?top_n=10
  - Returns collaborative-filtering recommendations for the given user_id.

The Flask app also exposes a POST /recommend form endpoint used by the UI.

## Data format
- data/items.csv — content items. Expected columns: content_id, title, description, sdg_goal, category
- data/users.csv — user metadata. Expected columns: user_id, name, age, country, interests
- data/interactions.csv — user-item interactions. Expected columns: user_id, content_id, rating, liked, viewed_time(min)

train_model.py performs a small cleaning step and exports cleaned CSVs into models/ for the app to load easily.

## Notes & Caveats
- The collaborative component uses TruncatedSVD on the user-item rating matrix (a compact approximation). It is lightweight but not as feature-rich as matrix factorization libraries.
- Ratings are clipped to the 1..5 range during training; outliers are handled conservatively.
- For unknown users, the system falls back to a popularity-based recommendation (most-interacted items).
- The hybrid scoring in recommender_utils.py uses a simple normalization assumption for collaborative scores — you may want to adjust normalization or apply better scaling depending on your production data.

## Extending or Improving
- Replace TruncatedSVD with a proper matrix factorization library (e.g., implicit, LightFM, or Surprise) for better CF performance.
- Add more item/user features and use neural or factorization methods to combine content and interaction data.
- Improve the hybrid strategy (e.g., per-user learned weights, rank aggregation).
- Add tests, CI, and Docker support for reproducible deployment.

## License & Contact
Feel free to reuse and adapt this code. If you'd like help extending it or deploying to production, open an issue or contact the maintainer.

Maintainer: aimlkhhushi1521 (GitHub)
