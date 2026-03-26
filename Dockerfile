FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY run.py       .
COPY config.yaml  .
COPY data.csv     .

# Run the pipeline with no hardcoded paths — all resolved inside the container
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
