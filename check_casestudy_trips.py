import pandas as pd
val = pd.read_parquet('validation_with_shap.parquet')

flagged = val[val['iso_forest_flag'] & val['has_reliable_route_baseline']]
top5 = flagged.sort_values('iso_forest_score', ascending=False).head(5)

print(top5[['PULocationID','DOLocationID','trip_distance','duration_minutes','speed_mph','fare_amount','fare_per_mile']].to_string())
print()
print("Key question for each: is trip_distance comfortably above 0.1, with a")
print("plausible duration? Or is it hugging the degenerate-trip boundary?")