import os
import time
import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === Config: Spotify API credential pairs ===
CREDENTIALS = [
    ("SPOTIPY_CLIENT_ID_1", "SPOTIPY_CLIENT_SECRET_1"),
    ("SPOTIPY_CLIENT_ID_2", "SPOTIPY_CLIENT_SECRET_2"),
]

# === Genre Mapping ===
GENRE_TO_CLASS = {
    "acoustic":     0,
    "folk":         0,
    "alt music":    1,
    "alternative":  1,
    "blues":        2,
    "bollywood":    3,
    "country":      4,
    "hip hop":      5,
    "hip-hop":      5,
    "indie":        6,
    "instrumental": 7,
    "metal":        8,
    "pop":          9,
    "rock":         10,
}

# === Helper: Get Spotify client ===
def init_spotify(index):
    os.environ["SPOTIPY_CLIENT_ID"] = os.getenv(CREDENTIALS[index][0])
    os.environ["SPOTIPY_CLIENT_SECRET"] = os.getenv(CREDENTIALS[index][1])
    manager = SpotifyClientCredentials()
    return Spotify(client_credentials_manager=manager)

# === Helper: Map genre tag to class ===
def map_tag_to_class(tag):
    tag = tag.lower()
    for key, val in GENRE_TO_CLASS.items():
        if key in tag:
            return val
    return None

# === Load test set ===
TEST_CSV    = "test.csv"
OUTPUT_CSV  = "test_with_genres.csv"
test_df     = pd.read_csv(TEST_CSV)

# === Init Spotify with first credential ===
sp_index = 0
sp = init_spotify(sp_index)

# === Process and fetch genres ===
results = []
interrupted = False

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Assigning genres"):
    artist = row["Artist Name"]
    track  = row["Track Name"]
    assigned = None
    retries = 0

    while retries < len(CREDENTIALS):
        try:
            query = f"track:{track} artist:{artist}"
            resp = sp.search(q=query, type="track", limit=1)
            items = resp.get("tracks", {}).get("items", [])
            if items:
                artist_id = items[0]["artists"][0]["id"]
                artist_info = sp.artist(artist_id)
                genres = artist_info.get("genres", [])
                for g in genres:
                    cls = map_tag_to_class(g)
                    if cls is not None:
                        assigned = cls
                        break
            break  # Success
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"⚠️ Rate limit hit on account {sp_index + 1}, switching...")
                retries += 1
                if retries < len(CREDENTIALS):
                    sp_index = (sp_index + 1) % len(CREDENTIALS)
                    sp = init_spotify(sp_index)
                    time.sleep(2)
                else:
                    print("❌ All API credentials exhausted. Saving progress...")
                    interrupted = True
                    break
            else:
                print(f"Error for '{artist} - {track}': {e}")
                break

    results.append(assigned if assigned is not None else -1)

    if interrupted:
        break

# === Save whatever progress we made ===
processed_df = test_df.iloc[:len(results)].copy()
processed_df["Class"] = results
partial_name = OUTPUT_CSV if not interrupted else OUTPUT_CSV.replace(".csv", "_partial.csv")
processed_df.to_csv(partial_name, index=False)

if interrupted:
    print(f"⚠️ Partial results saved to: {partial_name} ({len(results)} rows)")
else:
    print(f"✅ All results saved to: {OUTPUT_CSV}")
