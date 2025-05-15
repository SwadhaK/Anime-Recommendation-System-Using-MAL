import pickle
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import gdown

st.set_page_config(layout="wide")

@st.cache_data
def load_evaluation(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load all evaluations
evaluations = {
    "Normal Users": {
        "Content-Based Filtering": load_evaluation('data/cbf_evaluation'),
        "Collaborative Filtering": load_evaluation('data/cf_evaluation'),
        "Hybrid System": load_evaluation('data/hybrid_evaluation'),
    },
    "New Users": {
        "Content-Based Filtering": load_evaluation('data/new_cbf_evaluation'),
        "Collaborative Filtering": load_evaluation('data/new_cf_evaluation'),
        "Hybrid System": load_evaluation('data/new_hybrid_evaluation'),
    }
}

# Show results for selected user type
st.markdown("# 📊 Recommendation System Evaluation Results")
st.sidebar.markdown("## Select Evaluation Type")

user_type = st.selectbox("Select user type:", list(evaluations.keys()))

# Function to display metrics
def show_metrics(eval_dict, user_label):
    col1, col2 = st.columns(2)
    metrics = ['hit_rate', 'mean_reciprocal_rank']
    titles = {
        'hit_rate': f'Hit Rate Comparison ({user_label})',
        'mean_reciprocal_rank': f'Mean Reciprocal Rank Comparison ({user_label})'
    }

    for i, metric in enumerate(metrics):
        data = {
            "Recommender": [],
            "5 Recommendations": [],
            "10 Recommendations": [],
            "20 Recommendations": []
        }

        for name, eval in eval_dict.items():
            data["Recommender"].append(name)
            for k in ["5", "10", "20"]:
                data[f"{k} Recommendations"].append(eval[k][metric])

        df = pd.DataFrame(data)
        fig = px.bar(df, x="Recommender", y=df.columns[1:], barmode='group', title=titles[metric])

        col = col1 if metric == 'hit_rate' else col2
        col.markdown(f"### {metric.replace('_', ' ').title()}")
        if metric == 'hit_rate':
            col.markdown(
                "- **Hit Rate** measures the fraction of users for whom at least one relevant item was recommended."
                "<br>Higher is better (e.g., 0.25 = 25% of users got at least one relevant recommendation).",
                unsafe_allow_html=True)
        else:
            col.markdown(
                "- **MRR** indicates how early the first relevant recommendation appears."
                "<br>1.0 = always first, 0.5 = always second, etc.",
                unsafe_allow_html=True)

        col.plotly_chart(fig, use_container_width=True)
        col.markdown("#### 📋 Raw Data")
        col.dataframe(df)

        # Add best model info
        best_model = df.loc[df["10 Recommendations"].idxmax(), "Recommender"]
        best_score = df["10 Recommendations"].max()
        col.success(f"✅ Best model by 10 Recommendations: **{best_model}** ({metric.replace('_', ' ').title()}: {best_score:.3f})")

# Show the evaluation
show_metrics(evaluations[user_type], user_type)