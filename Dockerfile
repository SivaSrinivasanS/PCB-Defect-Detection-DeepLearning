# ✅ Use stable Python base
FROM python:3.9-slim-buster

# ✅ Set work directory
WORKDIR /app

# ✅ System dependencies
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

# ✅ Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Set model ID for gdown
ENV MODEL_FILE_ID=1pzSpYZgqHuDnVWt8u5j0vtUR_2UkXhyt

# ✅ Create models directory and download model via gdown
RUN mkdir -p models/ && \
    pip install gdown && \
    gdown --id ${MODEL_FILE_ID} --output models/pcb_cnn.h5

# ✅ List files for debug
RUN ls -lh models/

# ✅ Copy rest of the code
COPY . /app

# ✅ Streamlit port
EXPOSE 8501

# ✅ App startup command
CMD ["streamlit", "run", "src/pcb_ui.py"]
