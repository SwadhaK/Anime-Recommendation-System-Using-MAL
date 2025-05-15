# Anime Recommendation System Using MyAnimeList (MAL)

![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-brightgreen)
![Python](https://img.shields.io/badge/Python-3.2%2B-blue)

A powerful anime recommendation system that utilizes real user ratings from [MyAnimeList](https://myanimelist.net/). It supports content-based, collaborative, and hybrid filtering techniques, and is integrated with Streamlit for an interactive web interface.

🚀 **Live Demo**: [Try it on Streamlit](https://anime-recommendation-system-using-mal.streamlit.app/)

---

## 🔍 Features

- 🔗 Fetch user data using MAL username
- 🤖 Three recommendation approaches:
  - Content-Based Filtering
  - Collaborative Filtering (Matrix Factorization with SVD)
  - Hybrid (combining both)
- 📊 Displays recommendations in a clean tabular view
- 📦 Fully deployable via Streamlit Cloud

---

## 🧠 Recommendation Techniques

| Technique         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Content-Based     | Recommends based on anime features similar to those the user liked         |
| Collaborative     | Uses user-item interaction matrix with Surprise SVD for personalized suggestions |
| Hybrid            | Combines both methods for improved accuracy and coverage                   |

---

## 📁 Project Structure

```
anime-recommendation-system-using-mal/
├── Data/                     # Auto-downloaded data and model files
├── Web/
│   ├── pages/
│   │   └── Result.py         # Streamlit page logic
│   ├── Anime_Recommendation.py  # Main entry script for Streamlit
│   └── Recommendation_Systems_Core.py  # Core recommender classes
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/SwadhaK/Anime-Recommendation-System-Using-MAL.git
cd Anime-Recommendation-System-Using-MAL
```

### 2. Create a virtual environment & activate it

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run Web/Anime_Recommendation.py
```

---

## ☁️ Deployment (Streamlit Cloud)

This app is deployed on Streamlit Cloud.

📍 **Live App:** [https://anime-recommendation-system-using-mal.streamlit.app](https://anime-recommendation-system-using-mal.streamlit.app)


---

## 🔐 Environment & API Keys

To fetch user data via MAL, you may optionally set your `CLIENT_ID` in a file:

`Credentials.yml`
```yaml
api:
  client_id: your_mal_client_id_here
```

Or set it via environment variables (`CLIENT_ID`).

---

## ✨ Acknowledgments

- [MyAnimeList API](https://myanimelist.net/apiconfig/references/api/v2)
- [Surprise Library](https://surpriselib.com/) for collaborative filtering
- [Streamlit](https://streamlit.io) for rapid web deployment

