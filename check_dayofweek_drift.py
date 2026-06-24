import pandas as pd
df = pd.read_parquet('silver_taxi_features.parquet')

split_date = df['pickup_time'].quantile(0.83)
reference = df[df['pickup_time'] <= split_date]
current = df[df['pickup_time'] > split_date]

print("day_of_week distribution in REFERENCE period (Jan 1-26):")
print(reference['day_of_week'].value_counts(normalize=True).sort_index())
print()
print("day_of_week distribution in CURRENT period (Jan 26-31, only ~5 days):")
print(current['day_of_week'].value_counts(normalize=True).sort_index())
print()
print("If the current period is missing some days of the week entirely, or")
print("has them in wildly different proportions, that confirms the 2.52 PSI")
print("is a calendar-window artifact -- not real behavioral drift in the data.")
