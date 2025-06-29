import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import io
from PIL import Image
import gdown
import json

# --- DATABASE SETUP ---
DATABASE_URL = "sqlite:///pcb_database.db"
Base = declarative_base()

class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, default=datetime.now)
    image_path = Column(String)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def initialize_database():
    try:
        import sqlalchemy
        inspector = sqlalchemy.inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            st.success("✅ Table 'uploaded_images' ready.")
        else:
            st.warning("⚠️ Table not found.")
    except Exception as e:
        st.error(f"❌ DB init error: {e}")

# --- CLASS MAP LOADING ---
CLASS_MAP_PATH = "class_map.json"
def load_class_map():
    try:
        with open(CLASS_MAP_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Could not load class_map.json: {e}")
        return {}

CLASS_MAP = load_class_map()

# --- MODEL LOADING ---
MODEL_PATH = "models/pcb_cnn.h5"
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

@st.cache_resource
def load_and_prepare_model():
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found locally. Downloading...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model downloaded.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None

    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded.")
        return model
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None

# --- PREDICTION FUNCTION ---
def predict_defect(image_data, model):
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        probabilities = model.predict(img_array)[0]
        predicted_index = int(np.argmax(probabilities))
        label = CLASS_MAP.get(str(predicted_index), f"Class {predicted_index}")
        return label
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error"

# --- ADMIN TABLE VIEW ---
def show_upload_history():
    st.subheader("📁 Upload History")
    try:
        session = Session()
        records = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        session.close()
        if not records:
            st.info("No uploaded image records.")
            return
        data = {
            "Date": [r.datetime.strftime("%Y-%m-%d") for r in records],
            "Time": [r.datetime.strftime("%H:%M:%S") for r in records],
            "File": [os.path.basename(r.image_path) for r in records],
            "Path": [r.image_path for r in records],
        }
        st.dataframe(data, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error fetching history: {e}")

# --- STREAMLIT UI ---
def main():
    st.set_page_config(page_title="PCB Defect Classifier", layout="centered")
    st.title("🔍 PCB Board Defect Classifier")

    # Admin login section
    admin_logged_in = False
    with st.expander("🔐 Admin Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "PCB_Project" and password == "PCB123":
                st.success("✅ Admin logged in.")
                admin_logged_in = True
            else:
                st.error("❌ Invalid credentials.")

    model = load_and_prepare_model()
    uploaded_file = st.file_uploader("📤 Upload a PCB image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(image_data, caption="Uploaded Image", use_container_width=True)

        if model:
            st.info("⏳ Classifying...")
            result = predict_defect(image_data, model)
            st.write(f"### 🧠 Predicted Class: **{result}**")

            # Save to DB
            try:
                session = Session()
                os.makedirs("uploads", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"uploaded_image_{timestamp}_{uploaded_file.name}"
                save_path = os.path.join("uploads", filename)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                session.add(UploadedImage(image_path=save_path))
                session.commit()
                st.success("✅ Image and result saved.")
            except Exception as e:
                st.error(f"❌ Save error: {e}")
            finally:
                session.close()
        else:
            st.error("⚠️ Model not available.")

    # Show table if admin
    if admin_logged_in:
        show_upload_history()

# --- Run App ---
if __name__ == "__main__":
    initialize_database()
    main()
