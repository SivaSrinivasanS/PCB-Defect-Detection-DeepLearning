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

# --- Database Setup ---
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
    st.write("--- DATABASE INITIALIZATION STARTED ---")
    st.write(f"Checking/Creating database at: {DATABASE_URL}")
    try:
        import sqlalchemy
        inspector = sqlalchemy.inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            st.write("CONFIRMATION: Table 'uploaded_images' found in database.")
        else:
            st.error("ERROR: Table 'uploaded_images' not found after creation attempt.")
        st.write(f"Database tables created/checked successfully for engine at {DATABASE_URL}.")
    except Exception as e:
        st.error(f"Error initializing database: {e}")
    st.write("--- DATABASE INITIALIZATION FINISHED ---")

# --- Model Loading ---
MODEL_PATH = "models/pcb_cnn.h5"
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

@st.cache_resource
def load_and_prepare_model():
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Downloading...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None

    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None
    return model

# --- Prediction Function ---
def predict_defect(image_data, model):
    img = Image.open(io.BytesIO(image_data)).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    is_defective = predictions[0][0] > 0.5
    return "Defective" if is_defective else "Non-Defective", predictions[0][0]

# --- Streamlit UI ---
def main():
    st.title("PCB Board Class Predictor")

    with st.expander("Admin Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.success("Admin logged in.")
            else:
                st.error("Invalid credentials.")

    model = load_and_prepare_model()
    uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(image_data, caption="Uploaded Image", use_container_width=True)

        if model is not None:
            st.write("Classifying...")
            result, confidence = predict_defect(image_data, model)
            st.write(f"Prediction: **{result}** (Confidence: {confidence:.2f})")

            try:
                session = Session()
                os.makedirs("uploads", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"uploaded_image_{timestamp}_{uploaded_file.name}"
                path = os.path.join("uploads", image_filename)
                with open(path, "wb") as f:
                    f.write(image_data)
                session.add(UploadedImage(image_path=path))
                session.commit()
                st.success("Saved to database.")
            except Exception as e:
                st.error(f"DB save error: {e}")
            finally:
                session.close()
        else:
            st.error("Model not loaded.")

if __name__ == "__main__":
    initialize_database()
    main()
