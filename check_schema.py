import pandas as pd
df = pd.read_parquet('silver_taxi_features.parquet')
print("Columns in silver_taxi_features.parquet:")
for c in df.columns:
    print(f"  {c}  ({df[c].dtype})")
