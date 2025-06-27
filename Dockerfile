# Use Python 3.9, which has guaranteed compatibility with TensorFlow 2.10.0
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
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
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: DOWNLOAD PCB_CNN.H5 MODEL FROM GOOGLE DRIVE ---
ENV MODEL_FILE_ID "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"
RUN mkdir -p models/
RUN curl -L -o models/pcb_cnn.h5 "https://drive.google.com/uc?id=${MODEL_FILE_ID}&export=download"
RUN ls -lh models/ # Verify download in logs

# Copy the rest of your application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "src/pcb_ui.py"]
