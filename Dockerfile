# ✅ Dockerfile for PCB Defect Classifier with Python 3.9 + TensorFlow 2.10
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt ./

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements.txt

# Download model from Google Drive using gdown (already in requirements.txt)
RUN mkdir -p models
RUN python -m gdown.cli https://drive.google.com/uc?id=1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV -O models/pcb_cnn.h5 --fuzzy

# Copy rest of the codebase
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Entry point for the application
CMD ["streamlit", "run", "src/pcb_ui.py"]
