import os
import io
import math
import mmh3
from bitarray import bitarray
import pandas as pd
import great_expectations as ge
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from confluent_kafka import DeserializingConsumer, KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer

load_dotenv()

# --- SCALABLE BLOOM FILTER SETUP ---
class StreamingBloomFilter:
    def __init__(self, expected_elements=100000, false_positive_rate=0.01):
        self.m = int(- (expected_elements * math.log(false_positive_rate)) / (math.log(2) ** 2))
        self.k = int((self.m / expected_elements) * math.log(2))
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.k):
            self.bit_array[mmh3.hash(str(item), i) % self.m] = 1

    def contains(self, item):
        for i in range(self.k):
            if self.bit_array[mmh3.hash(str(item), i) % self.m] == 0:
                return False
        return True

bloom = StreamingBloomFilter(expected_elements=50000, false_positive_rate=0.01)

# --- STANDARD CONNECTOR CONFIGURATION ---
sr_conf = {
    'url': os.getenv('SCHEMA_REGISTRY_URL'),
    'basic.auth.user.info': f"{os.getenv('SCHEMA_REGISTRY_API_KEY')}:{os.getenv('SCHEMA_REGISTRY_API_SECRET')}"
}
schema_registry_client = SchemaRegistryClient(sr_conf)

consumer_conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': os.getenv('KAFKA_API_KEY'),
    'sasl.password': os.getenv('KAFKA_API_SECRET'),
    'key.deserializer': StringDeserializer('utf_8'),
    'value.deserializer': AvroDeserializer(schema_registry_client),
    'group.id': 'gcp-parquet-gx-group', # New group ID
    'auto.offset.reset': 'earliest'
}
consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe(['nyc-taxi-trips'])

storage_client = storage.Client()
bucket_name = os.getenv('GCS_BUCKET_NAME')
bucket = storage_client.bucket(bucket_name)

batch = []
duplicate_counts = 0

print("Lakehouse Consumer Running with Bloom Filter & Great Expectations Guard...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF: continue
            else: continue

        taxi_record = msg.value()
        
        # Deduplication Guard
        record_signature = f"{taxi_record.get('tpep_pickup_datetime')}_{taxi_record.get('total_amount')}"
        if bloom.contains(record_signature):
            duplicate_counts += 1
            if duplicate_counts % 50 == 0:
                print(f"[Deduplication Guard] Blocked {duplicate_counts} duplicates.")
            continue
        
        bloom.add(record_signature)
        batch.append(taxi_record)
        
        if len(batch) >= 100:
            df = pd.DataFrame(batch)
            
            # --- GREAT EXPECTATIONS DATA QUALITY CONTRACT ---
            ge_df = ge.from_pandas(df)
            
            # 1. Expect fares to be reasonably positive (no negative fares)
            fare_check = ge_df.expect_column_values_to_be_between(column="fare_amount", min_value=0.0, max_value=1000.0)
            
            # 2. Expect passenger counts to be realistic (0 to 10 max)
            pass_check = ge_df.expect_column_values_to_be_between(column="passenger_count", min_value=0, max_value=10)

            # If either expectation fails, quarantine the bad rows
            if not fare_check["success"] or not pass_check["success"]:
                print(f"⚠️ [Data Quality Alert] Bad records detected! Quarantining invalid rows...")
                
                # Filter out the bad rows to clean the dataframe before saving
                clean_df = df[(df['fare_amount'] >= 0.0) & (df['fare_amount'] <= 1000.0) & 
                              (df['passenger_count'] >= 0) & (df['passenger_count'] <= 10)]
            else:
                clean_df = df

            # Ensure we still have data left after filtering
            if clean_df.empty:
                print("❌ Entire batch failed quality checks. Dropping batch.")
                batch = []
                continue

            # --- UPLOAD CLEANED PARQUET DATA ---
            file_name = f"raw/taxi_data_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            blob = bucket.blob(file_name)
            
            parquet_buffer = io.BytesIO()
            clean_df.to_parquet(parquet_buffer, engine='pyarrow', index=False)
            
            blob.upload_from_string(data=parquet_buffer.getvalue(), content_type='application/octet-stream')
            print(f"✅ Success: Uploaded {len(clean_df)} validated records as PARQUET.")
            batch = []

except KeyboardInterrupt:
    print("\nStopping consumer gracefully...")
finally:
    consumer.close()