import pandas as pd
val = pd.read_parquet('validation_scored.parquet')

overall_rate = val['has_reliable_route_baseline'].eq(False).mean()
print(f'Overall rate of unreliable-baseline rows in validation set: {100*overall_rate:.2f}%')

flagged = val[val['iso_forest_flag']]
flagged_rate = flagged['has_reliable_route_baseline'].eq(False).mean()
print(f'Rate among flagged anomalies: {100*flagged_rate:.2f}%')

print()
print(f'Ratio (flagged rate / overall rate): {flagged_rate / overall_rate:.2f}x')