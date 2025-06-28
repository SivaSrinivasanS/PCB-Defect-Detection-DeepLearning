# ✅ Use Python 3.9 base - stable with TF 2.10.0
FROM python:3.9-slim-buster

# ✅ Set working directory
WORKDIR /app

# ✅ Install system dependencies
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

# ✅ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Set correct model ID (updated)
ENV MODEL_FILE_ID=1C3n4XFUsD6FiqcS79BA1pThGTcfEV0IG

# ✅ Create model directory and download model using gdown
RUN pip install gdown && \
    mkdir -p models && \
    gdown --id ${MODEL_FILE_ID} --output models/pcb_cnn.h5

# ✅ Confirm model is present (for logs)
RUN ls -lh models/

# ✅ Copy rest of your application code
COPY . /app

# ✅ Expose default Streamlit port
EXPOSE 8501

# ✅ Command to run Streamlit app
CMD ["streamlit", "run", "src/pcb_ui.py"]
