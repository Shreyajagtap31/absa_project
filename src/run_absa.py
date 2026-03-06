import os
import pandas as pd
from pyabsa.tasks.AspectSentimentTripletExtraction import AspectSentimentTripletExtractor

PROCESSED_DIR = os.path.expanduser("~/absa_project/data/processed")
OUTPUT_DIR    = os.path.expanduser("~/absa_project/output/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading ASTE model...")
extractor = AspectSentimentTripletExtractor(checkpoint="english", auto_device=False)
print("Model loaded.\n")

product_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
print(f"Found {len(product_files)} product files.\n")

for fname in sorted(product_files):
    asin = fname.replace(".csv", "")
    df   = pd.read_csv(os.path.join(PROCESSED_DIR, fname))

    # Column is now "text" (was reviewText before)
    text_col = next((c for c in ["text", "review_text", "reviewText", "review"]
                     if c in df.columns), None)
    if text_col is None:
        text_col = df.select_dtypes(include="object").columns[0]

    texts = df[text_col].dropna().astype(str).tolist()
    print(f"[{asin}] Processing {len(texts)} reviews...")

    rows = []
    for i, text in enumerate(texts):
        try:
            result   = extractor.predict(text, print_result=False)
            triplets = None
            if isinstance(result, dict):
                for key in ["Triplets", "triplets", "TRIPLETS", "aste_result"]:
                    if key in result and isinstance(result[key], list):
                        triplets = result[key]; break
                if triplets is None:
                    triplets = next((v for v in result.values() if isinstance(v, list)), None)
            elif isinstance(result, list):
                triplets = result

            for t in (triplets or []):
                aspect   = t.get("Aspect",   t.get("aspect",   "")).strip() if isinstance(t, dict) else str(t[0])
                opinion  = t.get("Opinion",  t.get("opinion",  "")).strip() if isinstance(t, dict) else str(t[1])
                polarity = t.get("Polarity", t.get("polarity", "")).strip() if isinstance(t, dict) else str(t[2])
                if aspect:
                    rows.append({"asin": asin, "review_idx": i,
                                 "aspect": aspect, "opinion": opinion,
                                 "polarity": polarity, "review_snippet": text[:120]})
        except Exception as e:
            print(f"  ⚠ review {i} skipped: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f"{asin}.csv"), index=False)
        print(f"  ✓ {len(rows)} triplets saved")
    else:
        print(f"  ✗ No triplets for {asin}")

print("\n✅ Done.")
