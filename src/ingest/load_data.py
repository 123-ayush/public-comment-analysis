import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(r"C:\Users\Ayush Ahlawat\OneDrive\Documents\Public Comment Analysis\public-comment-analysis\data\raw\policy_comments_10000.csv")

def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"{RAW_DATA_PATH} not found.")
    df = pd.read_csv(RAW_DATA_PATH)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded raw data shape:", df.shape)
