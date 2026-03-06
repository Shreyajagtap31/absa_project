import pandas as pd
import os

RAW_PATH      = os.path.expanduser("~/absa_project/data/raw/electronics_reviews.csv")
PROCESSED_DIR = os.path.expanduser("~/absa_project/data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
#   YOUR PRODUCTS — edit this list to add/remove products
# ============================================================
WANTED_ASINS = [
    # --- Original 9 (30+ reviews each) ---
    "B00TSUGXKE",   # Kindle Fire 7"
    "B00ZV9RDKK",   # Fire TV Stick
    "B010OYASRG",   # OontZ Angle 3 Speaker
    "B01DFKC2SO",   # Echo Dot 2nd Gen
    "B01MZEEFNX",   # Amazon Smart Plug
    "B0791TX5P5",   # Fire TV Stick Alexa
    "B079QHML21",   # Fire TV Stick 4K
    "B07FZ8S74R",   # Echo Dot 3rd Gen
    "B07HZLHPKP",   # Echo Show 5

    # --- New additions ---
    "B00OQVZDJM",   # Kindle Paperwhite 2015
    "B00CX5P8FC",   # Amazon Fire TV 1st Gen
    "B015TJD0Y4",   # Echo Dot 2nd Gen White
    "B07RGZ5NKS",   # TOZO T6 Wireless Earbuds
    "B08C1W5N87",   # Fire TV Stick 3rd Gen
    "B07J2Z5DBM",   # TOZO T10 Wireless Earbuds
    "B07VTK654B",   # Amazon Echo Auto
    "B07XJ8C8F5",   # Echo Dot 4th Gen
    "B003EM8008",   # Panasonic ErgoFit Earbuds
]
# ============================================================

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)
print(f"Total reviews in dataset: {len(df)}")

# Actual columns are: asin, text, rating, date
df = df[["asin", "text", "rating", "date"]].dropna(subset=["text"])
df = df[df["asin"].isin(WANTED_ASINS)]
print(f"Reviews found for your products: {len(df)}")

# Check which ASINs were NOT found
found     = df["asin"].unique().tolist()
not_found = [a for a in WANTED_ASINS if a not in found]
if not_found:
    print(f"\n⚠️  NOT found in dataset: {not_found}\n")

# Save one CSV per product
for asin, group in df.groupby("asin"):
    out_path = os.path.join(PROCESSED_DIR, f"{asin}.csv")
    group.to_csv(out_path, index=False)
    print(f"  ✓ {asin} — {len(group)} reviews")

print(f"\nDone. {len(found)} products saved to {PROCESSED_DIR}")
