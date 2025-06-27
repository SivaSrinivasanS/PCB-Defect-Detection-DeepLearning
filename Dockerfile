# Use a Python base image with a version known to be stable with TensorFlow 2.10.0.
# Python 3.9 is a great choice for this.
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install necessary system dependencies for various Python packages and file downloads.
# - curl: Used for downloading the model file.
# - build-essential: Needed for compiling some Python packages (e.g., related to SciPy/NumPy).
# - libgl1-mesa-glx, libxext6, libsm6, libxrender1: Common graphical libraries needed by some
#   data science/CV packages like OpenCV (if uncommented in requirements.txt).
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to keep the Docker image small

# Install Python dependencies from requirements.txt
# --no-cache-dir: Prevents pip from storing cached wheels, further reducing image size.
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL: DOWNLOAD PCB_CNN.H5 MODEL FROM GOOGLE DRIVE ---
# This environment variable holds the Google Drive File ID for direct download.
ENV MODEL_FILE_ID "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

# Create the 'models' directory where the pcb_cnn.h5 file will reside.
RUN mkdir -p models/

# Use curl to download the model file.
# -L: Follows HTTP redirects (Google Drive uses redirects for downloads).
# -o models/pcb_cnn.h5: Specifies the output file path and name inside the container.
# The URL pattern for direct Google Drive downloads: https://drive.google.com/uc?id=<FILE_ID>&export=download
RUN curl -L -o models/pcb_cnn.h5 "https://drive.google.com/uc?id=${MODEL_FILE_ID}&export=download"

# Optional: List the contents of the models/ directory to confirm the download in the build logs.
RUN ls -lh models/

# Copy the rest of your application code from your GitHub repository into the container.
# This includes src/, assets/, .github/, docs/, testing/, etc.
COPY . .

# Expose the default port for Streamlit applications.
EXPOSE 8501

# The command that Streamlit Cloud will run to start your application.
CMD ["streamlit", "run", "src/pcb_ui.py"]
