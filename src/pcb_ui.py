import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # Corrected import for load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import os
import io
from PIL import Image # Ensure Pillow is imported for image processing
import gdown # NEW: Import gdown for Google Drive downloads

# --- Database Setup (Existing Code) ---
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
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        session = Session()
        # Verify if tables exist
        inspector = sqlalchemy.inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            st.write("CONFIRMATION: Table 'uploaded_images' found in database.")
        else:
            st.error("ERROR: Table 'uploaded_images' not found after creation attempt.")
        session.close()
        st.write(f"Database tables created/checked successfully for engine at {DATABASE_URL}.")
    except Exception as e:
        st.error(f"Error initializing database: {e}")
    st.write("--- DATABASE INITIALIZATION FINISHED ---")

# --- Model Loading (Modified Code) ---
MODEL_PATH = "models/pcb_cnn.h5"
# Google Drive File ID for pcb_cnn.h5 (YOU CONFIRMED THIS ID)
# Make sure this file is publicly shareable for anyone with the link to view/download
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

@st.cache_resource # Use st.cache_resource for heavy objects like models
def load_and_prepare_model():
    model = None
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file not found at '{MODEL_PATH}'. Attempting to download from Google Drive...")
        try:
            # Use gdown to download the file directly
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            st.error("Please ensure the Google Drive link is publicly accessible and the file ID is correct.")
            return None # Return None if download fails

    try:
        model = load_model(MODEL_PATH)
        st.success("Deep Learning model loaded successfully!")
    except Exception as e:
        st.error(f"Cannot classify: Deep Learning model not loaded. Error: {e}")
        st.error("Please check the console for errors during model loading, and ensure 'train_model.ipynb' was run successfully to save the model.")
        return None
    return model

# --- Prediction Function (Existing Code) ---
def predict_defect(image_data, model):
    img = Image.open(io.BytesIO(image_data)).resize((224, 224)) # Resize to model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array /= 255.0 # Normalize image data

    predictions = model.predict(img_array)
    # Assuming binary classification, adjust if multi-class
    is_defective = predictions[0][0] > 0.5
    return "Defective" if is_defective else "Non-Defective", predictions[0][0]

# --- Streamlit UI (Existing Code with minor adjustment) ---
def main():
    st.title("PCB Board Class Predictor")

    # Call initialize_database early to ensure DB setup
    # initialize_database() # Commenting out or moving as st.write during cache messes with UI

    # Display an admin login section (as in previous discussions)
    with st.expander("Admin Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin": # Replace with secure credentials
                st.success("Admin logged in successfully!")
                # You might want to show admin-specific features here
            else:
                st.error("Invalid credentials.")

    # Load the model outside of prediction to ensure it's loaded once and cached
    model = load_and_prepare_model()

    uploaded_file = st.file_uploader("Upload an image of a PCB board to predict its class.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

        if model is not None:
            st.write("Classifying...")
            result, confidence = predict_defect(image_data, model)
            st.write(f"Prediction: **{result}** (Confidence: {confidence:.2f})")

            # Save uploaded image to database (existing code)
            try:
                session = Session()
                # Create a unique path for the uploaded image in the 'uploads' folder
                # Ensure 'uploads' directory exists
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"uploaded_image_{timestamp}_{uploaded_file.name}"
                image_full_path = os.path.join(upload_dir, image_filename)
                
                with open(image_full_path, "wb") as f:
                    f.write(image_data)
                
                new_image = UploadedImage(image_path=image_full_path)
                session.add(new_image)
                session.commit()
                st.success("Image uploaded and prediction performed.")
            except Exception as e:
                st.error(f"Error saving image to database: {e}")
            finally:
                session.close()
        else:
            st.error("Model not loaded. Cannot perform classification.")

if __name__ == "__main__":
    # Call database initialization once here.
    # It will run before Streamlit caches other functions if placed at module level.
    # Or, you can put this inside main() but make sure it doesn't print too much during re-runs.
    # For now, let's make it a general initialization call.
    initialize_database()
    main()

