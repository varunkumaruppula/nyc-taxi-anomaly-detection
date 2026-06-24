# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for confluent-kafka (C-based libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container
# NOTE: In a real production environment, we would inject secrets via CI/CD.
# For this local portfolio test, we are copying them directly.
COPY sink_script.py .
COPY .env .
COPY gcp-key.json .

# 6. Run the script when the container launches
CMD ["python", "sink_script.py"]