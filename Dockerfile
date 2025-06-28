# Use Python 3.9, which has guaranteed compatibility with TensorFlow 2.10.0
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install system dependencies first
COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: DOWNLOAD PCB_CNN.H5 MODEL FROM GOOGLE DRIVE ---
# This environment variable holds the Google Drive File ID for direct download.
ENV MODEL_FILE_ID "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

# Create the 'models' directory where the pcb_cnn.h5 file will reside.
RUN mkdir -p models/

# Use curl to download the model file directly from Google Drive
RUN curl -L -o models/pcb_cnn.h5 "https://drive.google.com/uc?id=${MODEL_FILE_ID}&export=download"

# Optional: List the contents of the models/ directory to confirm the download in the build logs.
RUN ls -lh models/

# Copy the rest of your application code
COPY . .

# Expose the default port for Streamlit applications.
EXPOSE 8501

# The command that Streamlit Cloud will run to start your application.
CMD ["streamlit", "run", "src/pcb_ui.py"]
