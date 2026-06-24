import pandas as pd
df = pd.read_parquet('silver_taxi_features.parquet')
print(list(df.columns))
print('has_reliable_route_baseline' in df.columns)
print('fare_per_mile_deviation_imputed' in df.columns)
print(df[['route_trip_count', 'fare_per_mile_deviation', 'has_reliable_route_baseline', 'fare_per_mile_deviation_imputed']].head(10))