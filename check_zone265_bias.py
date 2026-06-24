import pandas as pd
val = pd.read_parquet('validation_scored.parquet')

overall_265_rate = (val['DOLocationID'] == 265).mean()
flagged = val[val['iso_forest_flag']]
flagged_265_rate = (flagged['DOLocationID'] == 265).mean()

print(f'Overall rate of DOLocationID==265 in validation set: {100*overall_265_rate:.2f}%')
print(f'Rate among flagged anomalies: {100*flagged_265_rate:.2f}%')
print(f'Ratio: {flagged_265_rate/overall_265_rate if overall_265_rate > 0 else 0:.2f}x')

print()
print('Also checking PULocationID == 264 (Unknown) and either zone == 265 combined:')
overall_combo = (val['PULocationID'].isin([264,265]) | val['DOLocationID'].isin([264,265])).mean()
flagged_combo = (flagged['PULocationID'].isin([264,265]) | flagged['DOLocationID'].isin([264,265])).mean()
print(f'Overall rate (either zone is 264 or 265): {100*overall_combo:.2f}%')
print(f'Flagged rate (either zone is 264 or 265): {100*flagged_combo:.2f}%')
print(f'Ratio: {flagged_combo/overall_combo if overall_combo > 0 else 0:.2f}x')