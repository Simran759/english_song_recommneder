# preprocess.py
import os
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Set custom nltk_data path
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

# Define stopwords
stop_words = set(stopwords.words('english'))
tokenizer = WhitespaceTokenizer()  # Use whitespace tokenizer

# Text cleaning function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # Remove special characters and digits
    text = text.lower()  # Lowercase
    tokens = tokenizer.tokenize(text)  # Tokenize using whitespace
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
logging.info("🧼 Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text preprocessing completed.")

# TF-IDF vectorization
logging.info("🔠 Generating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity calculation
logging.info("📏 Computing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix)
logging.info("✅ Cosine similarity matrix computed.")

# Save outputs
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Preprocessed data saved successfully.")

logging.info("✅ All done! Preprocessing complete.")
