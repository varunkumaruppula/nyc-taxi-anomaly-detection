import duckdb
import gcsfs
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
bucket_name = os.getenv('GCS_BUCKET_NAME')

print("Authenticating with Google Cloud Storage...")

# 1. Point Google's official library to your JSON key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-key.json'

# 2. Initialize the file system and register it with DuckDB
fs = gcsfs.GCSFileSystem()
con = duckdb.connect()
con.register_filesystem(fs)

# 3. The Analytics Query
# Using read_parquet with union_by_name handles schema mismatches across streaming files
query = f"""
    SELECT 
        passenger_count,
        COUNT(*) as total_trips,
        ROUND(AVG(fare_amount), 2) as avg_fare,
        ROUND(AVG(trip_distance), 2) as avg_distance
    FROM read_parquet('gs://{bucket_name}/raw/*.parquet', union_by_name=True)
    GROUP BY passenger_count
    ORDER BY total_trips DESC;
"""

print(f"Executing remote SQL query across gs://{bucket_name}/raw/ ...")

# 4. Execute and fetch the results as a clean Pandas DataFrame
result_df = con.execute(query).df()

print("\n--- NYC Taxi Insights ---")
print(result_df)