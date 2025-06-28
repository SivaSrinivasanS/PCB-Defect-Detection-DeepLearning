import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image # Renamed to avoid conflict with PIL Image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import io
from PIL import Image
import gdown # Used for model download fallback if not handled by Dockerfile
import json # Import for JSON class mapping

# --- Configuration Constants ---
ADMIN_USERNAME = "TANSAM_TIDEL" # Your specified admin username
ADMIN_PASSWORD = "TANSAM123" # Your specified admin password

# --- Database Setup ---
Base = declarative_base()

# Determine the project root directory from the current script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Define the path for the SQLite database file in the project root
db_path = os.path.join(project_root_dir, 'pcb_database.db')

engine = create_engine(f'sqlite:///{db_path}', echo=False)

# Database model for uploaded images
class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String) # Stores the full path to the saved image file

# Use st.cache_resource to ensure database initialization runs only once
@st.cache_resource
def initialize_database():
    print(f"\n--- DATABASE INITIALIZATION STARTED ---")
    print(f"Checking/Creating database at: {db_path}")

    try:
        Base.metadata.create_all(engine)
        print(f"Database tables created/checked successfully for engine at {db_path}.")

        inspector = inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            print("CONFIRMATION: Table 'uploaded_images' found in database.")
        else:
            print("WARNING: Table 'uploaded_images' creation attempted, but not found.")
            st.warning("Table creation attempted, but not found. Check permissions or database path.")
            return False

        test_session = sessionmaker(bind=engine)()
        test_session.execute(text("SELECT 1"))
        test_session.close()
        print("Database connection successfully tested.")

    except Exception as e:
        print(f"CRITICAL ERROR during database initialization: {e}")
        st.error(f"A critical error occurred during database setup: {e}. "
                 "Please ensure the database file is not locked by another process "
                 "and that you have write permissions to the project directory.")
        return False

    print(f"--- DATABASE INITIALIZATION FINISHED ---")
    return True

if not initialize_database():
    st.stop()

Session = sessionmaker(bind=engine)

# --- Class Mapping Loading ---
class_map_path = os.path.join(project_root_dir, 'class_map.json')

@st.cache_resource
def load_class_map():
    try:
        if not os.path.exists(class_map_path):
            st.error(f"❌ Error: class_map.json not found at '{class_map_path}'. "
                     "Please ensure this file is created and placed in your project root "
                     "with the correct class index to label mappings.")
            return {}
        with open(class_map_path, "r") as f:
            class_map_data = json.load(f)
            return {str(k): v for k, v in class_map_data.items()} # Ensure keys are strings
    except json.JSONDecodeError:
        st.error(f"❌ Error: Could not decode class_map.json. Please check its format.")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading class_map.json: {e}")
        return {}

CLASS_MAP = load_class_map()
if not CLASS_MAP:
    st.warning("Class map could not be loaded. Predictions may show index numbers instead of labels.")


# --- Model Loading ---
MODEL_PATH = os.path.join(project_root_dir, 'models', 'pcb_cnn.h5')

@st.cache_resource
def load_and_prepare_model():
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found locally. Attempting to download from Google Drive...")
        GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            st.error("Please ensure the Google Drive link is publicly accessible and the file ID is correct.")
            return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading model from '{MODEL_PATH}': {e}")
        return None

classifier_model = load_and_prepare_model()

# --- PREDICTION FUNCTION ---
def predict_defect(image_data, model):
    if model is None:
        return "Error: Prediction model not available.", 0.0

    img_h, img_w = 224, 224

    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB").resize((img_h, img_w))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction_raw = model.predict(img_array, verbose=0)[0]
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(prediction_raw).numpy()
        
        predicted_index = int(np.argmax(probabilities))
        confidence = probabilities[predicted_index] # Use actual probability before rounding

        label = CLASS_MAP.get(str(predicted_index), f"Class {predicted_index} (Label Not Found)")

        return label, confidence

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error during prediction", 0.0

# --- STREAMLIT UI ---
def main():
    st.set_page_config(page_title="PCB Defect Classifier", layout="centered")
    st.title("🔍 PCB Board Defect Classifier")

    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False
    if "redirect" not in st.session_state:
        st.session_state["redirect"] = False

    admin_logged_in = st.session_state.get("admin_logged_in", False)

    if not admin_logged_in:
        if st.button("Admin Login", key="admin_login_button_main"):
            st.session_state["redirect"] = True
        
        if st.session_state.get("redirect", False):
            st.markdown("---")
            st.subheader("Admin Login")
            username = st.text_input("Username", key="admin_username")
            password = st.text_input("Password", type="password", key="admin_password")
            if st.button("Login", key="login_button_submit"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["redirect"] = False
                    st.success("✅ Admin logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")
        else:
            uploaded_file = st.file_uploader("📤 Upload a PCB image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_data = uploaded_file.read()
                st.image(image_data, caption="Uploaded Image", use_container_width=True)

                if classifier_model:
                    st.info("⏳ Classifying...")
                    result_label, confidence_score = predict_defect(image_data, classifier_model)
                    
                    st.write(f"### 🧠 Predicted Class: **{result_label.upper()}**")
                    st.write(f"**Confidence Score**: `{confidence_score:.4f}`")

                    if result_label == 'Non-Defective':
                        st.success("✅ This PCB is Non-Defective!")
                    elif result_label.startswith("Error"):
                        st.error("❌ Could not predict due to an error.")
                    else:
                        st.error(f"🔴 This PCB is DEFECTIVE: {result_label}")
                    
                    try:
                        session = Session()
                        uploads_dir = os.path.join(project_root_dir, "uploads")
                        os.makedirs(uploads_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = f"uploaded_image_{timestamp}_{uploaded_file.name}"
                        save_path = os.path.join(uploads_dir, image_filename)
                        
                        with open(save_path, "wb") as f:
                            f.write(image_data)
                            
                        new_image = UploadedImage(datetime=datetime.now(), image_path=save_path)
                        session.add(new_image)
                        session.commit()
                        st.success("✅ Image and result saved to database.")
                    except Exception as e:
                        st.error(f"❌ DB save error: {e}")
                    finally:
                        session.close()
                else:
                    st.error("⚠️ Model not available. Please check logs.")

    if admin_logged_in:
        st.markdown("---")
        st.subheader("Upload History (Admin View)")
        session = Session()
        history_query = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        session.close()

        history_display_data = []
        for item in history_query:
            history_display_data.append({
                "id": item.id,
                "Date": item.datetime.strftime("%Y-%m-%d"),
                "Time": item.datetime.strftime("%H:%M:%S"),
                "Image File": os.path.basename(item.image_path),
                "Full Path": item.image_path,
                "Delete": False
            })
        
        edited_df = st.data_editor(
            history_display_data,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Select to Delete",
                    help="Check to mark entries for deletion",
                    default=False,
                    required=False
                ),
                "id": None,
                "Full Path": None
            },
            hide_index=True,
            key="history_table_editor"
        )

        delete_ids = [item["id"] for item in edited_df if item["Delete"]]

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected Entries", key="delete_button") and delete_ids:
                session = Session()
                for img_id in delete_ids:
                    img_record = session.query(UploadedImage).filter_by(id=img_id).first()
                    if img_record and img_record.image_path and os.path.exists(img_record.image_path):
                        try:
                            os.remove(img_record.image_path)
                            print(f"Deleted image file from disk: {img_record.image_path}")
                        except Exception as file_e:
                            print(f"Error deleting file {img_record.image_path} from disk: {file_e}")
                            st.error(f"Could not delete file {os.path.basename(img_record.image_path)}. Please check permissions.")
                
                session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
                session.commit()
                session.close()
                st.success(f"Successfully deleted {len(delete_ids)} entries and associated files!")
                st.rerun()
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.session_state["redirect"] = False
                st.success("Logged out successfully!")
                st.rerun()
