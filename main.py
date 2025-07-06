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
                result = subprocess.run(["python", "preprocess.py"], check=True, capture_output=True, text=True)
                st.success("✅ Preprocessing completed successfully!")
                st.text(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("❌ Preprocessing failed!")
            st.code(e.stderr)
            st.stop()
        except Exception as e:
            st.error(f"🚫 Unexpected error: {e}")
            st.stop()

# Step 1: Ensure preprocessing is done
run_preprocessing_if_needed()

# Step 2: Now import after preprocessing
try:
    from recommend import df, recommend_songs
except Exception as e:
    st.error(f"❌ Failed to load recommend module: {e}")
    st.stop()

# Step 3: App UI
st.set_page_config(page_title="Music Recommender 🎵", page_icon="🎧", layout="centered")
st.title("🎶 Instant Music Recommender")

# Step 4: Song selection
if df is not None and 'song' in df.columns:
    song_list = sorted(df['song'].dropna().unique())
    selected_song = st.selectbox("🎵 Select a Song :", song_list)

    # Step 5: Recommend similar songs
    if st.button("🎙️ Recommend Similar Songs"):
        with st.spinner("🔍 Finding similar songs ..."):
            recommendation = recommend_songs(selected_song)
            if recommendation is None or recommendation.empty:
                st.warning("⚠️ Sorry, no similar songs found.")
            else:
                st.success("✅ Top Similar Songs:")
                st.table(recommendation)
else:
    st.error("🚫 Dataset not loaded properly or missing required columns.")
