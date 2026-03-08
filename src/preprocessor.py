import pandas as pd
import os

RAW_PATH      = os.path.expanduser("~/absa_project/data/raw/electronics_reviews.csv")
PROCESSED_DIR = os.path.expanduser("~/absa_project/data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WANTED_ASINS = [
    "B01G8JO5F2",   # NEW — 3,939 reviews
    "B013J7WUGC",   # NEW — 676 reviews
    "B00ZV9RDKK",   # Fire TV Stick — 1,091 reviews
    "B079QHML21",   # Fire TV Stick 4K — 831 reviews
    "B01DFKC2SO",   # Echo Dot 2nd Gen — 790 reviews
    "B07FZ8S74R",   # Echo Dot 3rd Gen — 552 reviews
    "B00TSUGXKE",   # Kindle Fire 7" — 543 reviews
    "B0791TX5P5",   # Fire TV Stick Alexa — 535 reviews
    "B010OYASRG",   # OontZ Angle 3 Speaker — 440 reviews
    "B01MZEEFNX",   # Amazon Smart Plug — 395 reviews
    "B00OQVZDJM",   # Kindle Paperwhite — 373 reviews
    "B07J2Z5DBM",   # TOZO T10 Earbuds — 322 reviews
    "B07HZLHPKP",   # Echo Show 5 — 321 reviews
    "B00CX5P8FC",   # Amazon Fire TV 1st Gen — 316 reviews
    "B003EM8008",   # Panasonic ErgoFit Earbuds — 262 reviews
    "B015TJD0Y4",   # Echo Dot 2nd Gen White — 255 reviews
    "B08C1W5N87",   # Fire TV Stick 3rd Gen — 227 reviews
    "B07RGZ5NKS",   # TOZO T6 Earbuds
    "B07VTK654B",   # Amazon Echo Auto
    "B07XJ8C8F5",   # Echo Dot 4th Gen
]

MAX_REVIEWS_PER_PRODUCT = 200

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)
print(f"Total reviews in dataset: {len(df):,}")

df = df[["asin", "text", "rating", "date"]].dropna(subset=["text"])
df = df[df["asin"].isin(WANTED_ASINS)]
print(f"Reviews found for your products: {len(df):,}")

found     = df["asin"].unique().tolist()
not_found = [a for a in WANTED_ASINS if a not in found]
if not_found:
    print(f"\n⚠️  NOT found: {not_found}\n")

for asin, group in df.groupby("asin"):
    group = group.head(MAX_REVIEWS_PER_PRODUCT)
    out_path = os.path.join(PROCESSED_DIR, f"{asin}.csv")
    group.to_csv(out_path, index=False)
    print(f"  ✓ {asin} — {len(group)} reviews saved")

total = sum(len(pd.read_csv(os.path.join(PROCESSED_DIR, f)))
            for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv"))
print(f"\nDone. {len(found)} products, {total:,} total reviews ready.")
