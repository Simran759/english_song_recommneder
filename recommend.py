# recommend.py
import joblib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading data...")
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
    bert_embeddings = joblib.load('bert_embeddings.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e


def recommend_songs(song_name, method="tfidf", top_n=5):
    """
    Recommend songs based on TF-IDF or BERT.
    :param song_name: Input song name
    :param method: "tfidf" or "bert"
    :param top_n: Number of recommendations
    :return: DataFrame of recommendations
    """
    logging.info("🎵 Recommending using '%s' for: '%s'", method.upper(), song_name)

    idx_list = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx_list) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None

    idx = idx_list[0]

    if method == "tfidf":
        sim_scores = list(enumerate(cosine_sim[idx]))
    elif method == "bert":
        sim_scores = list(enumerate(cosine_similarity([bert_embeddings[idx]], bert_embeddings)[0]))
    else:
        raise ValueError("Invalid method. Use 'tfidf' or 'bert'.")

    # Sort and get top_n similar songs excluding the input
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx][:top_n]
    song_indices = [i[0] for i in sim_scores]

    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index += 1
    result_df.index.name = "S.No."

    logging.info("✅ Recommendations generated using %s.", method.upper())
    return result_df
