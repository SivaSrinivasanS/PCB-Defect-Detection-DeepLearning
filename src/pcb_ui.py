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
            model_json = f.attrs.get('model_config')
            if model_json is None:
                st.error("❌ No model_config found in .h5 file.")
                return None
            model = model_from_json(model_json.decode('utf-8'))
            model.load_weights(MODEL_PATH)
        st.success("✅ Model loaded via model_from_json + weights.")
        return model
    except Exception as e:
        st.error(f"Error loading model manually: {e}")
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

# --- UI ---
def main():
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
            if uploaded_file:
                image_data = uploaded_file.read()
                st.image(image_data, caption="Uploaded Image", width=600)
                if classifier_model:
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
                        filename = f"uploaded_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
                        path = os.path.join(uploads_dir, filename)
                        with open(path, "wb") as f:
                            f.write(image_data)
                        session.add(UploadedImage(datetime=datetime.now(), image_path=path))
                        session.commit()
                        st.success("✅ Image and result saved to database.")
                    except Exception as e:
                        st.error(f"❌ DB save error: {e}")
                    finally:
                        session.close()
                else:
                    st.error("⚠️ Model not available. Please check logs.")

    if admin_logged_in:
        st.subheader("Upload History (Admin View)")
        session = Session()
        history = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        session.close()

        records = [
            {
                "id": h.id,
                "Date": h.datetime.strftime("%Y-%m-%d"),
                "Time": h.datetime.strftime("%H:%M:%S"),
                "Image File": os.path.basename(h.image_path),
                "Full Path": h.image_path,
                "Delete": False
            } for h in history
        ]

        edited_df = st.data_editor(records, column_config={
            "Delete": st.column_config.CheckboxColumn("Select to Delete", help="Mark entries for deletion"),
            "id": None,
            "Full Path": None
        }, hide_index=True, key="history_table_editor")

        delete_ids = [r["id"] for r in edited_df if r["Delete"]]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete Selected Entries", key="delete_button") and delete_ids:
                session = Session()
                for img_id in delete_ids:
                    img = session.query(UploadedImage).filter_by(id=img_id).first()
                    if img and img.image_path and os.path.exists(img.image_path):
                        try:
                            os.remove(img.image_path)
                        except Exception as file_e:
                            st.error(f"Could not delete file {img.image_path}.")
                session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
                session.commit()
                session.close()
                st.success(f"Deleted {len(delete_ids)} entries.")
                st.rerun()
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.session_state["redirect"] = False
                st.success("Logged out successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
