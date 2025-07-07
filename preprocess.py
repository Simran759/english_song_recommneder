# preprocess.py
import os
import pandas as pd
import re
import nltk
import joblib
import logging
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# ---------- File Check ----------
files_to_generate = {
    "df_cleaned.pkl": False,
    "tfidf_matrix.pkl": False,
    "cosine_sim.pkl": False,
    "bert_embeddings.pkl": False
}

for file in files_to_generate:
    files_to_generate[file] = not os.path.exists(file)

if not any(files_to_generate.values()):
    logging.info("✅ All preprocessing files already exist. Skipping.")
    exit(0)

# ---------- Load & Clean Dataset ----------
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000, random_state=42)
    df.drop(columns=['link'], inplace=True, errors='ignore')
    df.reset_index(drop=True, inplace=True)
    logging.info("✅ Loaded dataset with %d entries", len(df))
except Exception as e:
    logging.error("❌ Dataset load error: %s", str(e))
    raise

# ---------- Text Cleaning ----------
stop_words = set(stopwords.words('english'))
tokenizer = WhitespaceTokenizer()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

if files_to_generate["df_cleaned.pkl"] or any(files_to_generate.values()):
    logging.info("🧼 Cleaning lyrics...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    joblib.dump(df, "df_cleaned.pkl")
    logging.info("✅ Saved: df_cleaned.pkl")
else:
    df = joblib.load("df_cleaned.pkl")
    logging.info("📄 Loaded existing cleaned dataframe")

# ---------- TF-IDF ----------
if files_to_generate["tfidf_matrix.pkl"]:
    logging.info("🔡 Vectorizing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
    logging.info("✅ Saved: tfidf_matrix.pkl")
else:
    logging.info("📄 Skipped TF-IDF: already exists")

# ---------- Cosine Similarity ----------
if files_to_generate["cosine_sim.pkl"]:
    if not files_to_generate["tfidf_matrix.pkl"]:
        tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    logging.info("📏 Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    joblib.dump(cosine_sim, "cosine_sim.pkl")
    logging.info("✅ Saved: cosine_sim.pkl")
else:
    logging.info("📄 Skipped cosine similarity: already exists")

# ---------- BERT Embeddings ----------
if files_to_generate["bert_embeddings.pkl"]:
    logging.info("🤖 Generating BERT embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    joblib.dump(bert_embeddings, "bert_embeddings.pkl")
    logging.info("✅ Saved: bert_embeddings.pkl")
else:
    logging.info("📄 Skipped BERT embeddings: already exists")

logging.info("🎉 Preprocessing finished.")
