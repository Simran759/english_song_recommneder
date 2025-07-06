import streamlit as st
import os
import subprocess

# Run preprocessing if needed
def run_preprocessing_if_needed():
    required_files = ["df_cleaned.pkl", "tfidf_matrix.pkl", "cosine_sim.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.info("🔄 Preprocessing required. Please wait...")
        try:
            with st.spinner("Running preprocessing..."):
                subprocess.run(["python", "preprocess.py"], check=True)
            st.success("✅ Preprocessing completed successfully!")
        except Exception as e:
            st.error(f"❌ Failed to preprocess data: {e}")
            st.stop()

# Call preprocessing before importing recommend.py
run_preprocessing_if_needed()

# Now import the main logic
from recommend import df, recommend_songs

# App UI setup
st.set_page_config(page_title="Music Recommender 🎵", page_icon="🎧", layout="centered")

st.title("🎶 Instant Music Recommender")

# Song selection
song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a Song :", song_list)

# Recommendation trigger
if st.button("🎙️ Recommend Similar Songs"):
    with st.spinner("🔍 Finding similar songs ..."):
        recommendation = recommend_songs(selected_song)
        if recommendation is None:
            st.warning("⚠️ Sorry, song not found.")
        else:
            st.success("✅ Top Similar Songs:")
            st.table(recommendation)
