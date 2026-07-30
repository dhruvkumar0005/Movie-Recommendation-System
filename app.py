import os
import pickle
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #38bdf8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .movie-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
    }
    
    .movie-poster {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.4rem;
        height: 2.6rem;
        overflow: hidden;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    
    .badge-match {
        background: #10b981;
        color: #ffffff;
    }
    
    .badge-rating {
        background: #f59e0b;
        color: #ffffff;
    }
    
    .badge-genre {
        background: #3b82f6;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Load Pickled Models generated from Jupyter Notebook
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    movie_list_path = os.path.join(models_dir, 'movie_list.pkl')
    similarity_path = os.path.join(models_dir, 'similarity.pkl')
    
    if not os.path.exists(movie_list_path) or not os.path.exists(similarity_path):
        st.error("Model files not found! Please run the notebook/Movie_Recommendation_System.ipynb to generate the models.")
        st.stop()
        
    with open(movie_list_path, 'rb') as f:
        movies_df = pickle.load(f)
        
    with open(similarity_path, 'rb') as f:
        similarity_data = pickle.load(f)
        
    return movies_df, similarity_data

movies_df, similarity_data = load_models()

# Genre Poster Mapper
def get_poster(genre):
    posters = {
        'Action': 'https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=400&auto=format&fit=crop',
        'Adventure': 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=400&auto=format&fit=crop',
        'Animation': 'https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=400&auto=format&fit=crop',
        'Comedy': 'https://images.unsplash.com/photo-1514306191717-452ec28c7814?q=80&w=400&auto=format&fit=crop',
        'Drama': 'https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=400&auto=format&fit=crop',
        'Sci-Fi': 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=400&auto=format&fit=crop'
    }
    return posters.get(genre, 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=400&auto=format&fit=crop')

# Recommendation Logic
def recommend(movie_title, top_n=10):
    matched = movies_df[movies_df['title'] == movie_title]
    if matched.empty:
        return pd.DataFrame()
    idx = matched.index[0]
    
    top_indices = similarity_data['top_indices'][idx]
    top_scores = similarity_data['top_scores'][idx]
    
    recs = []
    for neighbor_idx, score in zip(top_indices, top_scores):
        if neighbor_idx == idx:
            continue
        row = movies_df.iloc[neighbor_idx].copy()
        row['similarity_score'] = round(float(score) * 100, 1)
        recs.append(row)
        if len(recs) >= top_n:
            break
    return pd.DataFrame(recs)

# Header UI
st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Select a movie to get instant content-based recommendations</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Number of Recommendations:", 1, 20, 10)

# Main Controls
movie_list = movies_df['title'].values
selected_movie = st.selectbox("Search / Select a Movie:", movie_list)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Finding similar movies..."):
        recs = recommend(selected_movie, top_n=top_n)
        
    if recs.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Recommendations for '{selected_movie}':")
        cols_per_row = 5
        for i in range(0, len(recs), cols_per_row):
            row_recs = recs.iloc[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, movie) in zip(cols, row_recs.iterrows()):
                with col:
                    genre = movie['genres_list'][0] if len(movie['genres_list']) > 0 else 'Drama'
                    poster_url = get_poster(genre)
                    st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" class="movie-poster" alt="{movie['title']}">
                        <div>
                            <div>
                                <span class="badge badge-match">{movie['similarity_score']}% Match</span>
                                <span class="badge badge-rating">★ {movie['avg_rating']}</span>
                            </div>
                            <div class="card-title">{movie['title']}</div>
                            <div>
                                {''.join([f'<span class="badge badge-genre">{g}</span>' for g in movie['genres_list'][:2]])}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    search_query = urllib.parse.quote(f"{movie['title']} trailer")
                    st.markdown(f"[🎬 Trailer](https://www.youtube.com/results?search_query={search_query})")
