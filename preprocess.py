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

# For BERT embeddings
from sentence_transformers import SentenceTransformer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# NLTK setup
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

logging.info("🚀 Preprocessing started...")

# Load and sample dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000, random_state=42)
    logging.info("✅ Dataset loaded successfully with %d samples", len(df))
except Exception as e:
    logging.error("❌ Error loading dataset: %s", str(e))
    raise

# Drop unnecessary columns
df.drop(columns=['link'], inplace=True, errors='ignore')
df.reset_index(drop=True, inplace=True)

# Stopwords and tokenizer
stop_words = set(stopwords.words('english'))
tokenizer = WhitespaceTokenizer()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Clean lyrics
logging.info("🧼 Cleaning lyrics...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaning complete.")

# TF-IDF vectorization
logging.info("🔠 Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity
logging.info("📏 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix)
logging.info("✅ Cosine similarity computed.")

# BERT embeddings
logging.info("🤖 Generating BERT embeddings...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    logging.info("✅ BERT embeddings shape: %s", bert_embeddings.shape)
except Exception as e:
    logging.error("❌ Failed to generate BERT embeddings: %s", str(e))
    raise

# Save all files
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
np.save('bert_embeddings.npy', bert_embeddings)
logging.info("💾 All files saved successfully.")

logging.info("🎉 Preprocessing complete.")
