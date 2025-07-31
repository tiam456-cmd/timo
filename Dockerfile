# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set display environment variable
ENV DISPLAY=:99

# Install system packages needed for GUI-based Python packages
RUN apt-get update && apt-get install -y \
    xvfb \
    libxrender1 libxext6 libsm6 libx11-6 \
    gcc \
    python3-dev \
    libevdev-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirments.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirments.txt

# Copy project files into the container
COPY . .

# Make sure the start script is executable
RUN chmod +x start.sh

# Expose the port used by uvicorn
EXPOSE 8000

# Start the app using Xvfb and uvicorn
CMD ["./start.sh"]
