import pandas as pd
val = pd.read_parquet('validation_scored.parquet')

flagged = val[val['iso_forest_flag']]
flagged_unreliable = flagged[~flagged['has_reliable_route_baseline']]
flagged_reliable = flagged[flagged['has_reliable_route_baseline']]

print(f"Flagged rows with UNRELIABLE baseline: {len(flagged_unreliable)}")
print(f"Flagged rows with RELIABLE baseline:   {len(flagged_reliable)}")
print()

print("Median speed_mph among flagged rows:")
print(f"  Unreliable baseline: {flagged_unreliable['speed_mph'].median():.2f}")
print(f"  Reliable baseline:   {flagged_reliable['speed_mph'].median():.2f}")
print()
print("Median fare_per_mile among flagged rows:")
print(f"  Unreliable baseline: {flagged_unreliable['fare_per_mile'].median():.2f}")
print(f"  Reliable baseline:   {flagged_reliable['fare_per_mile'].median():.2f}")
print()
print("Median baseline_zscore among flagged rows:")
print(f"  Unreliable baseline: {flagged_unreliable['baseline_zscore'].median():.2f}")
print(f"  Reliable baseline:   {flagged_reliable['baseline_zscore'].median():.2f}")
print()

overall_unreliable_rate = (~val['has_reliable_route_baseline']).mean()
baseline_flagged = val[val['baseline_flag']]
baseline_flagged_unreliable_rate = (~baseline_flagged['has_reliable_route_baseline']).mean()
print(f"Overall unreliable-baseline rate: {100*overall_unreliable_rate:.2f}%")
print(f"Rate among Z-SCORE BASELINE's own flagged rows: {100*baseline_flagged_unreliable_rate:.2f}%")
print(f"Ratio: {baseline_flagged_unreliable_rate/overall_unreliable_rate:.2f}x")