import streamlit as st
import os
import subprocess
import sys
import pandas as pd
from fpdf import FPDF
from io import BytesIO

# ---------- Step 1: Preprocessing ----------
def run_preprocessing_if_needed():
    required_files = ["df_cleaned.pkl", "tfidf_matrix.pkl", "cosine_sim.pkl","bert_embeddings.npy"]
    if not all(os.path.exists(file) for file in required_files):
        st.info("🔄 Preparing your music recommendations...")
        try:
            with st.spinner("🧠 Processing lyrics... This may take a moment."):
                result = subprocess.run(
                    [sys.executable, "preprocess.py"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            st.success("✅ Preprocessing complete!")
        except subprocess.CalledProcessError as e:
            st.error("❌ Failed to process data.")
            st.code(e.stderr)
            st.stop()
        except Exception as e:
            st.error(f"🚫 Unexpected error: {e}")
            st.stop()

run_preprocessing_if_needed()

# ---------- Step 2: Load Logic ----------
try:
    from recommend import df, recommend_songs
except Exception as e:
    st.error(f"❌ Error loading song data: {e}")
    st.stop()

# ---------- Step 3: App Layout ----------
st.set_page_config(page_title="🎧 Song Recommender", page_icon="🎵", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🎧 Song Matchmaker")
    st.markdown("Get personalized music recommendations based on lyrics 🎤.")

    model = st.radio("✨ Choose Your Match Style", [
        "🎯 Fast (Word Match)",
        "🧠 Smart (Meaning Match)"
    ])
    method = "tfidf" if "Fast" in model else "bert"

    selected_song = st.selectbox("🎵 Pick a song you like:", sorted(df['song'].dropna().unique()))

    top_n = st.slider("📊 How many recommendations?", min_value=3, max_value=15, value=5)

    if st.button("🔍 Find Songs Like This"):
        with st.spinner("🎼 Matching your vibe..."):
            results = recommend_songs(selected_song, method=method, top_n=top_n)
            st.session_state['recommendations'] = results
            st.session_state['selected_song'] = selected_song
            st.session_state['model'] = model

# ---------- Step 4: Display Recommendations ----------
st.markdown("""
    <h1 style='text-align: center;'>🎼 Discover Your Next Favorite Songs</h1>
    <p style='text-align: center;'>Pick one track and explore more songs with a similar vibe.</p>
""", unsafe_allow_html=True)
data=[]
if 'recommendations' in st.session_state:
    st.subheader(f"🎶 Songs similar to: *{st.session_state['selected_song']}*")
    st.caption(f"✨ Recommendation style: {st.session_state['model']}")

    results_df = st.session_state['recommendations']

    

    bg_color = st.get_option("theme.backgroundColor") or "#f9fbfc"
    text_color = st.get_option("theme.textColor") or "#000"

    for _, row in results_df.iterrows():
        song = row['song']
        artist = row['artist']
        lyrics = df[(df['song'] == song) & (df['artist'] == artist)]['text'].values
        lyrics_preview = lyrics[0][:300] + "..." if len(lyrics) > 0 else "Lyrics not available"
        youtube_url = f"https://www.youtube.com/results?search_query={song.replace(' ', '+')}+{artist.replace(' ', '+')}"

        data.append({
        "Song": song,
        "Artist": artist,
        "Lyrics Preview": lyrics_preview,
        "YouTube Link": youtube_url
        })

        st.markdown(f"""
        <div style="
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
        ">
        <h4 style="margin-bottom: 5px;">🎵 <strong>{song}</strong></h4>
        <p style="margin: 0 0 10px;">👤 <strong>{artist}</strong></p>
        <p style="font-size: 14px;"><em>{lyrics_preview}</em></p>
        <a href="{youtube_url}" target="_blank" style="
            display: inline-block;
            padding: 8px 14px;
            background-color: #e63946;
            color: white;
            border-radius: 6px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
        ">▶️ Listen on YouTube</a>
        </div>
        """, unsafe_allow_html=True)


    # ---------- Step 5: Download as CSV or PDF ----------
 # ---------- Step 5: Download as CSV or PDF ----------
export_df = pd.DataFrame(data)

col1, col2 = st.columns([1, 1])
with col1:
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Download as CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

with col2:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for index, row in export_df.iterrows():
        pdf.cell(200, 10, txt=f"{index + 1}. {row['Song']} - {row['Artist']}", ln=True)
        pdf.multi_cell(0, 8, f"Lyrics Preview: {row['Lyrics Preview']}")
        pdf.cell(0, 10, txt=f"Link: {row['YouTube Link']}", ln=True)
        pdf.ln(5)

    buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Convert to bytes
    buffer.write(pdf_bytes)
    buffer.seek(0)

    st.download_button("📕 Download as PDF", data=buffer, file_name="recommendations.pdf", mime="application/pdf")
