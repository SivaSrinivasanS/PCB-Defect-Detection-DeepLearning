FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./

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

RUN mkdir -p models

RUN python -m gdown.cli https://drive.google.com/uc?id=1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV -O models/pcb_cnn.h5 --fuzzy

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/pcb_ui.py"]