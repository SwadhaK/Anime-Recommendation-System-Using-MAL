import os
import pickle
import streamlit as st
import gdown
import pandas as pd
import yaml

from Recommendation_Systems_Core import *

# 1. Setup Streamlit Page
st.set_page_config(layout="wide")
st.title("🎌 Anime Recommender System")
st.sidebar.header("📋 Choose Recommendation Settings")

# 2. Ensure data directory exists
os.makedirs("Data", exist_ok=True)

# 3. Load client ID if present (not actively used)
def load_client_id(config_path='Credentials.yml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['api']['client_id']
    return os.getenv('CLIENT_ID')

client_id = load_client_id()

# 4. Define recommender models and Google Drive file IDs
RECOMMENDER_FILES = {
    "Content-Based Filtering": {
        "id": "1Hncv-TqPr-IrEYzC1gppfPWS_sXdSgXJ",
        "path": "Data/cbf_recommender"
    },
    "Collaborative Filtering": {
        "id": "18s2OpjB39TsURcT_K9P9kP2Gi3zyipxB",
        "path": "Data/collaborative_recommender"
    },
    "Hybrid": {
        "id": "1KmnwKrMhN-enh6s7fiosXzc8mxLfWZUY",
        "path": "Data/hybrid_recommender"
    }
}

# 5. Cache + lazy load recommender
@st.cache_data(show_spinner=False)
def load_recommender_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    with open(output_path, 'rb') as f:
        return pickle.load(f)

# 6. Recommendation Logic Per Type
def get_cbf_recommendations(r, user_id, n):
    return r.recommend_user(user_id, n)

def get_cf_recommendations(r, user_id, n):
    internal_id = r.cf_trainset.to_inner_uid(user_id)
    raw_results = r.recommend_user(internal_id, n)
    return r.get_user_recommendations_df(raw_results)

def get_hybrid_recommendations(r, user_id, n):
    raw_results = r.recommend_user(user_id, n)
    return r.get_user_anime_recommendations_df(raw_results)

RECOMMENDATION_FUNCTIONS = {
    "Content-Based Filtering": get_cbf_recommendations,
    "Collaborative Filtering": get_cf_recommendations,
    "Hybrid": get_hybrid_recommendations
}

# 7. UI Inputs
recommender_type = st.sidebar.selectbox(
    "Select Recommendation Type",
    list(RECOMMENDER_FILES.keys()),
    index=0
)

username = st.sidebar.text_input("Enter MyAnimeList Username")
n_recommendations = st.sidebar.number_input("Number of Recommendations", min_value=1, value=5)

# 8. Trigger and Output
if st.sidebar.button("🔍 Get Recommendations"):
    if not username.strip():
        st.warning("Please enter a valid MyAnimeList username.")
        st.stop()

    with st.status("Loading model...", expanded=True) as status:
        try:
            model_info = RECOMMENDER_FILES[recommender_type]
            recommender = load_recommender_from_drive(model_info["id"], model_info["path"])
            status.update(label="Model loaded successfully ✅", state="complete")
        except Exception as e:
            st.error(f"Failed to load recommender model: {e}")
            st.stop()

    with st.spinner("Fetching recommendations..."):
        try:
            user_id = recommender.add_new_user_by_mal_username(username)
            recommendations = RECOMMENDATION_FUNCTIONS[recommender_type](recommender, user_id, n_recommendations)

            if "synopsis" in recommendations:
                recommendations["synopsis"] = recommendations["synopsis"].str.slice(stop=100) + "..."

            for col in ["combined", "distance"]:
                if col in recommendations.columns:
                    recommendations.drop(columns=[col], inplace=True)

            st.success(f"Here are your {n_recommendations} recommendations:")
            st.dataframe(recommendations)
        except Exception as e:
            st.error(f"Failed to get recommendations: {e}")









