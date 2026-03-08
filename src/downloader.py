from huggingface_hub import hf_hub_download
import json
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading Electronics reviews...")

filepath = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/Electronics.jsonl",
    repo_type="dataset"
)

print(f"File cached at: {filepath}")
print("Parsing 500,000 reviews...")

reviews = []
with open(filepath, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500000:
            break
        r = json.loads(line)
        reviews.append({
            "asin":   r.get("asin", ""),
            "text":   r.get("text", ""),
            "rating": r.get("rating", 0),
            "date":   r.get("timestamp", "")
        })
        if i % 50000 == 0:
            print(f"  Parsed {i:,}...")

df = pd.DataFrame(reviews)
df = df[df["text"].str.len() > 50]
df.to_csv("data/raw/electronics_reviews.csv", index=False)

print(f"\nDone! {len(df):,} reviews saved.")
print(f"Unique ASINs: {df['asin'].nunique():,}")
print("\nTop 20 ASINs by review count:")
print(df["asin"].value_counts().head(20))
