import pandas as pd
import difflib
import re
from pathlib import Path

districts = [
    'Jaipur','Jodhpur','Udaipur','Alwar','Kolkata','Bengaluru','Bengaluru Urban','Chennai',
    'Mumbai','Pune','Lucknow','Varanasi','Agra','Kanpur','Patna','Bhopal','Hyderabad',
    'Visakhapatnam','Ahmedabad','Surat','Guwahati','Imphal','Kohima','Raipur',
    'Thiruvananthapuram'
]

def extract_exact(text):
    if not isinstance(text, str): return None
    t = text.lower()
    for d in districts:
        if d.lower() in t:
            return d
    return None

def extract_fuzzy(text):
    if not isinstance(text, str): return None
    match = difflib.get_close_matches(text, districts, n=1, cutoff=0.75)
    return match[0] if match else None

def fill_districts(df):
    df["district_filled"] = df["district"]

    # 1. Exact match from location
    mask = df["district_filled"].isna() | (df["district_filled"] == "")
    df.loc[mask, "district_filled"] = df.loc[mask, "location_free_text"].apply(extract_exact)

    # 2. Exact match from comment text
    mask = df["district_filled"].isna() | (df["district_filled"] == "")
    df.loc[mask, "district_filled"] = df.loc[mask, "text"].apply(extract_exact)

    # 3. Fuzzy match from location
    mask = df["district_filled"].isna() | (df["district_filled"] == "")
    df.loc[mask, "district_filled"] = df.loc[mask, "location_free_text"].apply(extract_fuzzy)

    # 4. Fuzzy match from text
    mask = df["district_filled"].isna() | (df["district_filled"] == "")
    df.loc[mask, "district_filled"] = df.loc[mask, "text"].apply(extract_fuzzy)

    return df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Ayush Ahlawat\OneDrive\Documents\Public Comment Analysis\public-comment-analysis\data\raw\policy_comments_10000.csv")
    df = fill_districts(df)
    df.to_csv(r"C:\Users\Ayush Ahlawat\OneDrive\Documents\Public Comment Analysis\public-comment-analysis\data\processed\comments_geo_resolved.csv", index=False)
    print("Geo-resolution complete.")
