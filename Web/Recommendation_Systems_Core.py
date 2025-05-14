import pandas as pd
import numpy as np
import yaml
import requests
import time
import os
import gdown
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

# from surprise import BaselineOnly, Dataset, Reader, accuracy, SVD, SVDpp, NMF
# from surprise.model_selection import GridSearchCV, train_test_split

os.makedirs("Data", exist_ok=True)

# Replace these with actual file IDs from Drive
file_map = {
    "anime.csv": "1Qr2MnAJGZoy3LTrYhnh0qA9_B2DQtv-0",
    "user_ratings_train.csv": "11VX_8qR0RZTPXPywYq2naG7sXLQtN_v0",
    "user_ratings_test.csv": "1yT9-VW-wFrKnwErEckQSomPWjN3Y1E99",
    "new_user_ratings_train.csv": "1R-YA3_BiOmtcTnxDzDKpDQZGRg37zJQY",
    "new_user_ratings_test.csv": "1hBRwHnOvwyP-vY79vz7d5gylIuMXnXRb"
}

for filename, file_id in file_map.items():
    output_path = os.path.join("Data", filename)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} already exists. Skipping.")


""" Read file"""
anime_df = pd.read_csv('./Data/anime.csv')
train_df = pd.read_csv('./Data/user_ratings_train.csv')
test_df = pd.read_csv('./Data/user_ratings_test.csv')
new_users_train_df = pd.read_csv('./Data/new_user_ratings_train.csv')
new_users_test_df = pd.read_csv('./Data/new_user_ratings_test.csv')

for user_id in new_users_train_df.user_id.unique():
    num_anime_watched = len(new_users_test_df[new_users_test_df.user_id == user_id])
    if num_anime_watched == 0:
        print(user_id, " watched no anime")
        break


"""Setup the Distance Functions"""
# cosine distances
def cosine_distance_metric(user_profile, anime_vector_df):
    user_vector_df = np.array([user_profile["weighted_vector_avg"]])
    dist_matrix = cosine_distances(user_vector_df, anime_vector_df).flatten()
    return dist_matrix

# euclidean distances
def euclidean_distances_metric(user_profile, anime_vector_df):
    user_vector_df = np.array([user_profile["weighted_vector_avg"]])
    dist_matrix = euclidean_distances(user_vector_df, anime_vector_df).flatten()
    return dist_matrix

# manhattan distances
def manhattan_distances_metric(user_profile, anime_vector_df):
    user_vector_df = np.array([user_profile["weighted_vector_avg"]])
    dist_matrix = manhattan_distances(user_vector_df, anime_vector_df).flatten()
    return dist_matrix

""" Client Id"""
def load_client_id(config_path = '../Credentials.yml'):
    if os.path.exists(config_path):
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
        return config['api']['client_id']
    else:
        return os.getenv('CLIENT_ID')



""" Adding new user's list from MyAnimeList to User Rating"""

CLIENT_ID = load_client_id()  # ID to authenticate requests to the API

MY_ANIME_LIST_API_URL = 'https://api.myanimelist.net/v2'  # The API endpoint that will be accessed.
API_REQUEST_DELAY = 1  # Delay to prevent over-use of the API

def get_users_list(username: str, new_user_id: int):
    """Collects anime lists and appends to an existing JSON file"""
    response: requests.Response = None
    try:
        response = requests.get(f"{MY_ANIME_LIST_API_URL}/users/{username}/animelist?fields=list_status&limit=1000",
                                headers={'X-MAL-CLIENT-ID': CLIENT_ID})
        time.sleep(API_REQUEST_DELAY * 2)

        user_anime_list = {"username": username, "anime_list": []}
        anime_list_data = response.json().get('data')
        if anime_list_data:
            for node in anime_list_data:
                user_anime_list["anime_list"].append({"anime": node["node"], "list_status": node["list_status"]})

        while response.json().get('paging', {}).get('next'):
            time.sleep(API_REQUEST_DELAY * 2)
            response = requests.get(response.json()['paging']['next'] + "&fields=list_status",
                                    headers={'X-MAL-CLIENT-ID': CLIENT_ID})
            anime_list_data = response.json().get('data')
            if anime_list_data:
                for node in anime_list_data:
                    user_anime_list["anime_list"].append(
                        {"anime": node["node"], "list_status": node["list_status"]})

        print(f"Found {username}'s anime list. They watched {len(user_anime_list['anime_list'])} anime...")

    except Exception as e:
        print(f"Exception occurred while processing {username}: {e}")

    user_ratings_dict = {
        "user_id": [],
        "anime_id": [],
        "score": []
    }

    for anime_rating in user_anime_list["anime_list"]:
        if anime_rating["list_status"].get("status") == "completed" and anime_df.id.isin(
                [anime_rating["anime"]["id"]]).any():
            user_ratings_dict["user_id"].append(new_user_id)
            user_ratings_dict["anime_id"].append(anime_rating["anime"]["id"])
            user_ratings_dict["score"].append(anime_rating["list_status"]["score"])

    print("Finished writing user anime lists.")

    return pd.DataFrame(data=user_ratings_dict)


# get next user id
def get_next_user_id(user_ratings_df: pd.DataFrame):
    next_user_id = max(user_ratings_df.user_id.unique()) + 1
    return next_user_id


"""Create Recommender System"""

"""Content-Based Filtering Recommender"""

class CBFRecommender:
    def __init__(self, anime_data: pd.DataFrame, user_ratings_data: pd.DataFrame):
        self.anime_df = anime_data
        self.user_ratings_data = user_ratings_data
        self.users_to_ignore_on_evaluation = []
        self.custom_user_mappings = {}  # username: user_id

    # TF-IDF Vectorization of anime data
    def vectorize_anime_data(self, stop_words='english', max_features=50, max_df=0.5, min_df=0.01):
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features, max_df=max_df, min_df=min_df)
        vectors = vectorizer.fit_transform(self.anime_df['genres'])
        self.anime_vector_df = pd.DataFrame(vectors.toarray(), index=self.anime_df['id'])
        return self.anime_vector_df

    # Build the user's taste profile
    def create_user_profile(self, user_id: int):
        user_ratings = self.user_ratings_data[self.user_ratings_data['user_id'] == user_id]
        if user_ratings.empty:
            return None  # No ratings available for this user
        high_ratings = user_ratings[user_ratings.score >= user_ratings.score.mean()]
        anime_vectors = self.anime_vector_df.loc[self.anime_vector_df.index.isin(high_ratings['anime_id'])]
        if anime_vectors.empty:
            return None  # No vectors available for the rated anime
        return anime_vectors.mean()

    # Calculate distance between user profile and all anime
    def get_user_anime_distance(self, user_id: int, distance_metric='cosine'):
        user_profile = self.create_user_profile(user_id)

        metrics = {
            'cosine': cosine_distances,
            'euclidean': euclidean_distances,
            'manhattan': manhattan_distances
        }
        distance_func = metrics.get(distance_metric)
        if distance_func is None:
            raise ValueError(f'Invalid distance metric: {distance_metric}')
        distances = distance_func([user_profile], self.anime_vector_df.values).flatten()
        distance_df = pd.DataFrame({'anime_id': self.anime_vector_df.index, 'distance': distances})

        watched_ids = self.user_ratings_data[self.user_ratings_data.user_id == user_id]['anime_id']
        return distance_df[~distance_df.anime_id.isin(watched_ids)].sort_values('distance')

    # Recommend top N anime to user
    def recommend_user(self, user_id: int, num_recommendations=10, distance_metric='cosine', add_anime_info=True):
        recommendations = self.get_user_anime_distance(user_id, distance_metric).head(num_recommendations)
        if add_anime_info:
            recommendations = recommendations.merge(self.anime_df, left_on='anime_id', right_on='id')
        return recommendations

    # Return anime with their distance scores
    def get_user_anime_scores(self, user_id: int, distance_metric='cosine'):
        scores = self.get_user_anime_distance(user_id, distance_metric)
        return scores.rename(columns={'distance': 'score'})

    # Add a new user using a MyAnimeList username
    def add_new_user_by_mal_username(self, username: str):
        if username in self.custom_user_mappings:
            return self.custom_user_mappings[username]

        user_id = get_next_user_id(self.user_ratings_data)
        list_df = get_users_list(username, user_id)
        self.user_ratings_data = pd.concat([self.user_ratings_data, list_df])
        self.users_to_ignore_on_evaluation.append(user_id)
        self.custom_user_mappings[username] = user_id
        return user_id

    # Evaluate hit rate and mean reciprocal rank
    def evaluate(self, num_recommendations: int, testing_df: pd.DataFrame, distance_metric='cosine'):
        users = testing_df['user_id'].unique()
        hits, reciprocal_ranks = 0, []

        for user_id in users:
            if user_id in self.users_to_ignore_on_evaluation:
                continue
            recommendations = self.recommend_user(user_id, num_recommendations, distance_metric, add_anime_info=False)
            if recommendations.empty:
                continue  # Skip users with no recommendations

            recommendations = self.recommend_user(user_id, num_recommendations, distance_metric, add_anime_info=False)
            test_anime_ids = testing_df[testing_df.user_id == user_id]['anime_id'].values
            recommend_anime_ids = recommendations['anime_id'].values

            hit_positions = [i for i, anime_id in enumerate(recommend_anime_ids) if anime_id in test_anime_ids]
            if hit_positions:
                hits += 1
                reciprocal_ranks.append(1 / (hit_positions[0] + 1))

        total_users = len(users)
        return {
            "hit_rate": hits / total_users,
            "mean_reciprocal_rank": np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        }

    # Make recommendations for all the users
    def make_recommendations(self, testing_df: pd.DataFrame, num_recommendations=20, distance_metric='cosine'):
        users = testing_df['user_id'].unique()  # FIXED: was wrongly typed as "testings_df"
        recs = {}

        for user_id in users:
            actual_animes = self.anime_vector_df.loc[
                self.anime_vector_df.index.isin(testing_df[testing_df.user_id == user_id]['anime_id'])
            ]
            if len(actual_animes) > 0:
                predictions = self.recommend_user(user_id, num_recommendations, distance_metric, add_anime_info=False)
                prediction_vectors = self.anime_vector_df.loc[
                    self.anime_vector_df.index.isin(predictions['anime_id'])
                ]
                recs[user_id] = {
                    "actual_anime_vectors": actual_animes.to_numpy(),
                    "predictions": predictions,
                    "prediction_vectors": prediction_vectors.to_numpy()
                }
            else:
                print(f"Warning: Not enough anime vectors for user {user_id}")

        return recs




# convert user ratings DataFrame into a format compatible with the Surprise library for collaborative filtering
def convert_user_ratings_to_surprise_dataset(ur_df):
    surprise_user_rating_df = ur_df[["user_id", "anime_id", "score"]]
    surprise_user_rating_df = surprise_user_rating_df.rename(columns={
        "user_id": "userID",
        "anime_id": "itemID",
        "score": "rating"
    })

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(surprise_user_rating_df, reader)
    return data


"""Collaborative Filtering Recommender"""

class CollaborativeFilteringRecommender:
    def __init__(self, ratings_dataset: pd.DataFrame, mf_algorithm=SVD):
        # Save the ratings and algorithm
        self.ratings_dataset = ratings_dataset
        self.mf_algorithm = mf_algorithm      # matrix factorization model to use (default is SVD).

        # User tracking
        self.custom_user_mappings = {}
        self.users_to_ignore_on_evaluation = []

        # Train model
        self._fit_model()

    def _fit_model(self):
        # Convert ratings to Surprise format and train the model
        data = convert_user_ratings_to_surprise_dataset(self.ratings_dataset)
        self.cf_trainset = data.build_full_trainset()
        self.cf_model = self.mf_algorithm()
        self.cf_model.fit(self.cf_trainset)

    def _get_unrated_items(self, inner_user_id):
        # Get items the user hasn't rated yet
        rated = {iid for (iid, _) in self.cf_trainset.ur[inner_user_id]}
        return [iid for iid in self.cf_trainset.all_items() if iid not in rated]

    def get_user_anime_scores(self, user_id: str):
        # Predict scores for all anime the user hasn't rated yet
        inner_id = self.cf_trainset.to_inner_uid(user_id)
        predictions = [
            self.cf_model.predict(user_id, self.cf_trainset.to_raw_iid(iid))
            for iid in self._get_unrated_items(inner_id)
        ]
        return pd.DataFrame({
            'id': [int(p.iid) for p in predictions],
            'score': [p.est for p in predictions]
        })

    def recommend_user(self, inner_user_id, num_recommendations):
        # Recommend top N anime for a user
        user_id = self.cf_trainset.to_raw_uid(inner_user_id)
        predictions = [
            self.cf_model.predict(user_id, self.cf_trainset.to_raw_iid(iid))
            for iid in self._get_unrated_items(inner_user_id)
        ]
        top_n = sorted(predictions, key=lambda p: p.est, reverse=True)[:num_recommendations]
        return [int(p.iid) for p in top_n]

    def add_new_user_by_mal_username(self, username: str):
        # Add new user ratings and retrain model
        if username in self.custom_user_mappings:
            return self.custom_user_mappings[username]

        user_id = get_next_user_id(self.ratings_dataset)
        user_ratings = get_users_list(username, user_id)

        self.custom_user_mappings[username] = user_id
        self.users_to_ignore_on_evaluation.append(user_id)
        self.ratings_dataset = pd.concat([self.ratings_dataset, user_ratings])

        print("Refitting model...")
        self._fit_model()
        print("Model retrained.")

        return user_id

    def evaluate(self, num_recommendations: int, testing_df: pd.DataFrame):
        # Evaluate using Hit Rate and Mean Reciprocal Rank (MRR)
        hits, mrr_sum, tested_users = 0, 0, 0

        for inner_id in self.cf_trainset.all_users():
            user_id = int(self.cf_trainset.to_raw_uid(inner_id))
            if user_id in self.users_to_ignore_on_evaluation:
                continue

            recommendations = self.recommend_user(inner_id, num_recommendations)
            actual = testing_df[testing_df.user_id == user_id]['anime_id'].tolist()

            for rank, anime_id in enumerate(recommendations):
                if anime_id in actual:
                    hits += 1
                    mrr_sum += 1 / (rank + 1)
                    break

            tested_users += 1

        return {
            "hit_rate": hits / tested_users if tested_users else 0,
            "mean_reciprocal_rank": mrr_sum / tested_users if tested_users else 0
        }

    def get_user_recommendations_df(self, user_recommendations: list):
        # Convert recommended anime IDs to full anime details
        return anime_df[anime_df.id.isin(user_recommendations)]

    def make_recommendations(self, num_recommendations, test_ratings_df, anime_vector_df):
        # Return recommendation vectors for each user
        recommendations = {}

        for inner_id in self.cf_trainset.all_users():
            user_id = int(self.cf_trainset.to_raw_uid(inner_id))
            actual_ids = test_ratings_df[test_ratings_df.user_id == user_id]['anime_id']

            actual_vectors = anime_vector_df.loc[anime_vector_df.index.isin(actual_ids)]
            predicted_ids = self.recommend_user(inner_id, num_recommendations)
            predicted_vectors = anime_vector_df.loc[anime_vector_df.index.isin(predicted_ids)]

            if predicted_vectors.empty:
                print(f"No predicted anime vectors found for user {user_id}.")

            recommendations[user_id] = {
                "actual_anime_vectors": actual_vectors.to_numpy(),
                "predictions": predicted_ids,
                "prediction_vectors": predicted_vectors.to_numpy()
            }

        return recommendations


"""Create the Hybrid Recommender"""

class HybridRecommender:
    def __init__(self, anime_data, user_ratings_data, cbf_distance_metric='cosine'):
        # Initialize content-based and collaborative recommenders
        self.anime_data = anime_data
        self.user_ratings_data = user_ratings_data
        self.cbf_distance_metric = cbf_distance_metric
        self.users_to_ignore_on_evaluation = []

        self.cbf_recommender = CBFRecommender(anime_data, user_ratings_data)
        self.cbf_recommender.vectorize_anime_data()

        self.cf_recommender = CollaborativeFilteringRecommender(user_ratings_data)

    def get_user_combined_scores(self, user_id):
        # Get content-based scores
        cbf_scores = self.cbf_recommender.get_user_anime_scores(user_id, distance_metric=self.cbf_distance_metric)
        cbf_scores.rename(columns={"score": "cbf_score", "anime_id": "id"}, inplace=True)

        # Get collaborative filtering scores
        cf_scores = self.cf_recommender.get_user_anime_scores(user_id)
        cf_scores["score"] = (cf_scores["score"] - cf_scores["score"].min()) / (cf_scores["score"].max() - cf_scores["score"].min())
        cf_scores["cf_score"] = 1 - cf_scores["score"]
        cf_scores.drop(columns=["score"], inplace=True)

        # Merge the two sets of scores
        combined = cbf_scores.merge(cf_scores, on='id', how='inner')
        combined['combined_score'] = combined['cbf_score'] + combined['cf_score']

        # Lower score = better match
        return combined.sort_values(by='combined_score')

    def recommend_user(self, user_id, num_recommendations):
        # Return top N recommendations for a user
        scores = self.get_user_combined_scores(user_id)
        return scores.head(num_recommendations)

    def add_new_user_by_mal_username(self, username):
        # Add a new user to both systems
        cbf_id = self.cbf_recommender.add_new_user_by_mal_username(username)
        cf_id = self.cf_recommender.add_new_user_by_mal_username(username)

        if cbf_id != cf_id:
            print(f"Warning: Mismatched user IDs ({cbf_id} != {cf_id})")

        self.users_to_ignore_on_evaluation.append(cf_id)
        return cf_id

    def get_user_anime_recommendations_df(self, recommendations_df):
        # Return full anime info for given recommendations
        return anime_df[anime_df.id.isin(recommendations_df.id)]

    def evaluate(self, num_recommendations, test_df):
        # Evaluate model using Hit Rate and Mean Reciprocal Rank
        hits = 0
        mrr_sum = 0
        user_ids = test_df['user_id'].unique()

        for user_id in user_ids:
            if user_id in self.users_to_ignore_on_evaluation:
                continue

            recommendations = self.recommend_user(user_id, num_recommendations)
            actual = test_df[test_df.user_id == user_id]['anime_id'].values

            for rank, anime_id in enumerate(recommendations.id):
                if anime_id in actual:
                    hits += 1
                    mrr_sum += 1 / (rank + 1)
                    break

        total = len(user_ids)
        return {
            "hit_rate": hits / total if total else 0,
            "mean_reciprocal_rank": mrr_sum / total if total else 0
        }

    def make_recommendations(self, num_recommendations, test_df, anime_vector_df):
        # Generate recommendations for all users
        results = {}

        for user_id in self.user_ratings_data['user_id'].unique():
            recs = self.recommend_user(user_id, num_recommendations)
            actual = test_df[test_df.user_id == user_id]['anime_id']

            results[user_id] = {
                "actual_anime_vectors": anime_vector_df.loc[anime_vector_df.index.isin(actual)].to_numpy(),
                "predictions": recs,
                "prediction_vectors": anime_vector_df.loc[anime_vector_df.index.isin(recs['id'])].to_numpy()
            }

        return results



