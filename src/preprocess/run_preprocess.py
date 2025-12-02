import pandas as pd
from clean_text import clean_comment

df = pd.read_csv("data/processed/comments_geo_resolved.csv")

# Fix: replace NaN in original text
df["text"] = df["text"].fillna("").astype(str)

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_comment)

# Fix: clean_text must not have NaN
df["clean_text"] = df["clean_text"].fillna("").astype(str)

df.to_csv("data/processed/comments_cleaned.csv", index=False)
print("Preprocessing completed successfully.")
