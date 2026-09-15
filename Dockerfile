FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (just in case, though opencv-python-headless doesn't need libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (HF Spaces uses 7860 by default for Docker spaces, but Flask defaults to 5000. 
# We can tell gunicorn to bind to 0.0.0.0:7860)
EXPOSE 7860

# Run gunicorn (Listen on PORT env variable provided by Render/Koyeb)
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --timeout 120 app_tflite:app
