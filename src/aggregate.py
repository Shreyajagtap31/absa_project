import os
import pandas as pd

RESULTS_DIR = os.path.expanduser("~/absa_project/output/results")
OUTPUT_DIR  = os.path.expanduser("~/absa_project/output")

# ============================================================
#   PRODUCT NAMES — all 20 products
# ============================================================
PRODUCT_NAMES = {
    "B01G8JO5F2": "Amazon Tap Speaker",
    "B013J7WUGC":  "Fire HD 8 Tablet",
    "B00ZV9RDKK":  "Fire TV Stick",
    "B079QHML21":  "Fire TV Stick 4K",
    "B01DFKC2SO":  "Echo Dot 2nd Gen",
    "B07FZ8S74R":  "Echo Dot 3rd Gen",
    "B00TSUGXKE":  "Kindle Fire 7\"",
    "B0791TX5P5":  "Fire TV Stick Alexa",
    "B010OYASRG":  "OontZ Angle 3 Speaker",
    "B01MZEEFNX":  "Amazon Smart Plug",
    "B00OQVZDJM":  "Kindle Paperwhite",
    "B07J2Z5DBM":  "TOZO T10 Earbuds",
    "B07HZLHPKP":  "Echo Show 5",
    "B00CX5P8FC":  "Amazon Fire TV 1st Gen",
    "B003EM8008":  "Panasonic ErgoFit Earbuds",
    "B015TJD0Y4":  "Echo Dot 2nd Gen White",
    "B08C1W5N87":  "Fire TV Stick 3rd Gen",
    "B07RGZ5NKS":  "TOZO T6 Earbuds",
    "B07VTK654B":  "Amazon Echo Auto",
    "B07XJ8C8F5":  "Echo Dot 4th Gen",
}
# ============================================================

dfs = []
for fname in os.listdir(RESULTS_DIR):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
        dfs.append(df)

if not dfs:
    print("❌ No result CSVs found. Run run_absa.py first.")
    exit()

master = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(master):,} total triplets")

# Normalize polarity
master["polarity"] = master["polarity"].str.strip().str.lower()
master["polarity"] = master["polarity"].map({
    "pos": "Positive", "positive": "Positive",
    "neg": "Negative", "negative": "Negative",
    "neu": "Neutral",  "neutral":  "Neutral",
}).fillna(master["polarity"].str.capitalize())

# Normalize aspect
master["aspect"] = master["aspect"].str.lower().str.strip()

# Add product name
master["product_name"] = master["asin"].map(PRODUCT_NAMES).fillna(master["asin"])

# Save master
master.to_csv(os.path.join(OUTPUT_DIR, "master_triplets.csv"), index=False)
print(f"✓ master_triplets.csv saved ({len(master):,} rows)")

# Summary stats
summary = master.groupby(["asin", "product_name", "aspect", "polarity"]).size().unstack(fill_value=0)
for col in ["Positive", "Negative", "Neutral"]:
    if col not in summary.columns:
        summary[col] = 0
summary["total"]           = summary[["Positive", "Negative", "Neutral"]].sum(axis=1)
summary["sentiment_score"] = summary["Positive"] - summary["Negative"]
summary = summary.reset_index().sort_values("total", ascending=False)
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)
print(f"✓ summary_stats.csv saved ({len(summary):,} rows)")

print("\nTop 10 most-mentioned aspects:")
print(summary.groupby("aspect")["total"].sum().sort_values(ascending=False).head(10).to_string())
