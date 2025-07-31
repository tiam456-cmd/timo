# Use official Python 3.10 slim image
# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set display environment variable
ENV DISPLAY=:99

# Install system packages needed for GUI-based Python packages and build dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    procps \
    libxrender1 libxext6 libsm6 libx11-6 \
    gcc \
    python3-dev \
    libevdev-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy project files into the container
COPY . .

# Expose default uvicorn port for Render and others
EXPOSE 8000

# Use PORT env variable with fallback to 8000 for local runs
CMD bash -c "\
  Xvfb :99 -screen 0 1024x768x24 & \
  sleep 2 && \
  if pgrep Xvfb > /dev/null; then echo '✅ Xvfb is running on DISPLAY=:99'; else echo '❌ Xvfb failed to start' && exit 1; fi && \
  export DISPLAY=:99 && \
  PORT=${PORT:-8000} && \
  echo \"🚀 Starting FastAPI server on port $PORT...\" && \
  exec uvicorn gpt:app --host 0.0.0.0 --port $PORT \
"
