# Use the official Python 3.11 slim image to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for ML libraries (e.g., scipy, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the Docker image size smaller
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the data directory exists so the app doesn't crash on upload
RUN mkdir -p data/eeg_scans

# Expose port 5000 for the Flask API
EXPOSE 5000

# Set the default port for Gunicorn
ENV FLASK_PORT=5000

# Run the unified Flask API using Gunicorn for production
# We set a high timeout (120s) because parsing EEG files can be slow
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "api.main:app"]
