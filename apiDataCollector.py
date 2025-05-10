"""
This script retrieves data for all available anime (up to 500 entries per request) and saves it into a JSON file.

It also gathers usernames — 20 per requested page — which can later be used to fetch each user's anime list individually.
"""

import requests
import json
import time
import yaml
import os

# Client id
def load_client_id(config_path = '../Credentials.yml'):
    if os.path.exist(config_path):
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
        return config['api']['client_id']
    else:
        return os.getenv('CLIENT_ID')


CLIENT_ID = load_client_id()  # ID to authenticate request sts to the API
print(f'Client ID: {CLIENT_ID}')

MY_ANIME_LIST_API_URL = 'https://api.myanimelist.net/v2' # The API endpoint that will be accessed.
API_REQUEST_DELAY = 1 # Delay to prevent over-use of the API


# Collect Anime data
def collect_anime_data():
    anime_rankings_request_queries = 'limit=500&fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics'

    collected_anime = []
    is_next_page_available = True
    current_url = f'{MY_ANIME_LIST_API_URL}/anime/ranking?{anime_rankings_request_queries}'
    while is_next_page_available:
        anime_rankings_request: requests.Response = None
        try:
            anime_rankings_request = requests.get(current_url, headers={'X-MAL-CLIENT-ID': CLIENT_ID})
            time.sleep(API_REQUEST_DELAY)

            for node in anime_rankings_request.json()['data']:
                collected_anime.append(node['node'])
            
            next_page_url = anime_rankings_request.json()['paging'].get('next')
            if next_page_url is None:
                is_next_page_available = False
            else:
                current_url = next_page_url
                print("Collecting next page of anime...")
        except:
            print(f'Exception occured while attempting to make the request: {anime_rankings_request.json()} from the anime rankings endpoint.')

    print("Finished collecting anime...")
    return {
        "anime": collected_anime
    }

# Collect User data
def collect_usernames(num_pages=100):
    usernames = []

    for page_num in range(1, num_pages+1):
        users_request: requests.Response = None
        try:
            users_request = requests.get(f'https://api.jikan.moe/v4/users?page={page_num}')
            time.sleep(API_REQUEST_DELAY)
            
            for user in users_request.json()['data']:
                usernames.append(user['username'])
            
            print("Collected next page of usernames...")
        except:
            print(f"Exception occured when trying to perform user request: {users_request.json()}")

    print("Collected all usernames..")

    return {
        "usernames": usernames
    }


def get_last_processed_username(output_file):
    """Finds the last successfully processed username from the output file."""
    try:
        with open(output_file, 'r') as f:
            data = f.read().strip()
            if data and data != "[":
                json_data = json.loads(data.rstrip(",\n]") + "]")  # Ensure valid JSON
                return json_data[-1]["username"] if json_data else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return None


def collect_anime_lists(json_file, output_file):
    """Collects anime lists and appends to an existing JSON file"""

    # Load usernames from source file
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        usernames = json_data["usernames"]
    print(f'Num usernames found: {len(usernames)}')

    # Find the last processed username to resume
    last_processed = get_last_processed_username(output_file)
    start_index = usernames.index(last_processed) + 1 if last_processed in usernames else 0
    print(f"Resuming from index {start_index} ({usernames[start_index]}).")

    # Open file in append mode, ensuring proper JSON formatting
    with open(output_file, 'a') as f:
        if start_index == 0:
            f.write("[\n")  # Start JSON array only if the file is empty
        else:
            f.seek(0, 2)  # Move to end of file
            f.write("\n")  # Ensure a new line for appending

    for ind, username in enumerate(usernames[start_index:], start=start_index + 1):
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

            # Append the new data
            with open(output_file, 'a') as f:
                json.dump(user_anime_list, f, indent=4)
                f.write(",\n")

            print(f"{ind}. Found {username}'s anime list. They watched {len(user_anime_list['anime_list'])} anime...")

        except Exception as e:
            print(f"Exception occurred while processing {username}: {e}")
            break  # Stop processing if an error occurs to avoid corrupting JSON

    # Properly close JSON array
    with open(output_file, 'rb+') as f:
        f.seek(-2, 2)  # Remove the last comma
        f.truncate()
        f.write(b'\n]')

    print("Finished writing user anime lists.")


def write_json_to_file(data, filename):
    json_data = json.dumps(data, indent=4)

    with open(filename, "w") as json_file:
        json_file.write(json_data)

if __name__ == '__main__':
    # Collect Data on Every Anime Available on MyAnimeList
    write_json_to_file(collect_anime_data(), 'data/anime.json')

    # Collect a List of Usernames
    write_json_to_file(collect_usernames(num_pages=500), 'data/usernames.json')

    # Collect a List of User Anime List Data (Requires data/usernames.json file)
    collect_anime_lists('data/usernames.json', 'data/user_anime_lists.json')
