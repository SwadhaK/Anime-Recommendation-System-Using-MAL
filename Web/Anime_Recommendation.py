import pickle
import streamlit as st
import gdown
import pandas as pd
import yaml
import os


# Load client ID from config
def load_client_id(config_path = '../Credentials.yml'):
    if os.path.exists(config_path):
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
        return config['api']['client_id']
    else:
        return os.getenv('CLIENT_ID')


client_id = load_client_id()

from Recommendation_Systems_Core import *

st.set_page_config(layout="wide")
st.markdown("# Get Recommendations For MyAnimeList Accounts")
st.sidebar.markdown("# Recommendations For MyAnimeList Accounts")


os.makedirs("Data", exist_ok=True)

# Replace these with actual file IDs from Drive
file_map = {
    "cbf_recommender": "1Hncv-TqPr-IrEYzC1gppfPWS_sXdSgXJ",
    "collaborative_recommender": "18s2OpjB39TsURcT_K9P9kP2Gi3zyipxB",
    "hybrid_recommender": "1KmnwKrMhN-enh6s7fiosXzc8mxLfWZUY"
}

for filename, file_id in file_map.items():
    output_path = os.path.join("Data", filename)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} already exists. Skipping.")


# Load recommenders
@st.cache_data
def load_recommender(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

recommenders = {
    "Content-Based Filtering Recommender": load_recommender('./Data/cbf_recommender'),
    "Collaborative Filtering Recommender": load_recommender('./Data/collaborative_recommender'),
    "Hybrid Recommender": load_recommender('./Data/hybrid_recommender')
}

# UI container
container = st.container(border=True)
recommender_option = container.selectbox(
    "Which recommender would you like to use?",
    list(recommenders.keys()),
    index=0
)

username = container.text_input("Enter MyAnimeList Username")
num_recommendations_input = container.text_input("Enter Number of Recommendations", value="5")

try:
    n_recommendations = max(1, int(num_recommendations_input))
except ValueError:
    st.warning("Number of recommendations must be an integer.")
    st.stop()

selected_recommender = recommenders[recommender_option]

# Define logic for showing recommendations
def display_results(recommender, recommend_fn):
    try:
        user_id = recommender.add_new_user_by_mal_username(username)
        recommendations = recommend_fn(recommender, user_id, n_recommendations)

        if "synopsis" in recommendations:
            recommendations["synopsis"] = recommendations["synopsis"].str.slice(stop=100) + "..."

        for col in ["combined", "distance"]:
            if col in recommendations.columns:
                recommendations.drop(columns=[col], inplace=True)

        st.table(recommendations)
    except Exception as e:
        st.error(f"Failed to get recommendations: {e}")

# Recommender-specific logic
def get_cbf_recommendations(r, user_id, n):
    return r.recommend_user(user_id, n)

def get_cf_recommendations(r, user_id, n):
    internal_id = r.cf_trainset.to_inner_uid(user_id)
    raw_results = r.recommend_user(internal_id, n)
    return r.get_user_recommendations_df(raw_results)

def get_hybrid_recommendations(r, user_id, n):
    raw_results = r.recommend_user(user_id, n)
    return r.get_user_anime_recommendations_df(raw_results)

recommendation_functions = {
    "Content-Based Filtering Recommender": get_cbf_recommendations,
    "Collaborative Filtering Recommender": get_cf_recommendations,
    "Hybrid Recommender": get_hybrid_recommendations
}

# Trigger on button click
if container.button("Get Recommendations", type="primary"):
    display_results(selected_recommender, recommendation_functions[recommender_option])









