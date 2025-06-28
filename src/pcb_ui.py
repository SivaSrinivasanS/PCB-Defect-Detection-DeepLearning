import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import io
from PIL import Image
import gdown
import json
import h5py

st.set_page_config(page_title="PCB Defect Classifier", layout="centered")

# --- Configuration Constants ---
ADMIN_USERNAME = "PCB_project"
ADMIN_PASSWORD = "PCB123"

# --- Database Setup ---
Base = declarative_base()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
db_path = os.path.join(project_root_dir, 'pcb_database.db')
engine = create_engine(f'sqlite:///{db_path}', echo=False)

class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String)

@st.experimental_singleton
def initialize_database():
    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        if 'uploaded_images' not in inspector.get_table_names():
            st.warning("Table creation attempted, but not found. Check permissions or database path.")
            return False
        test_session = sessionmaker(bind=engine)()
        test_session.execute(text("SELECT 1"))
        test_session.close()
    except Exception as e:
        st.error(f"A critical error occurred during database setup: {e}.")
        return False
    return True

if not initialize_database():
    st.stop()

Session = sessionmaker(bind=engine)

# --- Class Mapping ---
class_map_path = os.path.join(project_root_dir, 'class_map.json')

@st.experimental_singleton
def load_class_map():
    try:
        if not os.path.exists(class_map_path):
            st.error(f"❌ Error: class_map.json not found at '{class_map_path}'.")
            return {}
        with open(class_map_path, "r") as f:
            return {str(k): v for k, v in json.load(f).items()}
    except Exception as e:
        st.error(f"❌ Error loading class_map.json: {e}")
        return {}

CLASS_MAP = load_class_map()
if not CLASS_MAP:
    st.warning("Class map could not be loaded. Predictions may show index numbers instead of labels.")

# --- Model Loading ---
MODEL_PATH = os.path.join(project_root_dir, 'models', 'pcb_cnn.h5')

@st.experimental_singleton
def load_and_prepare_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found locally. Attempting to download from Google Drive...")
        try:
            gdown.download(id="1C3n4XFUsD6FiqcS79BA1pThGTcfEV0IG", output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                st.error("❌ No model_config found in .h5 file.")
                return None
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')

        model = model_from_json(model_config)
        model.load_weights(MODEL_PATH)
        st.success("✅ Model loaded using JSON config + weights.")
        return model
    except Exception as e:
        st.error(f"❌ Manual model loading failed: {e}")
        return None

classifier_model = load_and_prepare_model()

# --- Prediction ---
def predict_defect(image_data, model):
    if model is None:
        return "Error: Prediction model not available.", 0.0
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        probabilities = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        label = CLASS_MAP.get(str(predicted_index), f"Class {predicted_index} (Label Not Found)")
        return label, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error during prediction", 0.0
