import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image # Renamed to avoid conflict with PIL Image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import io
from PIL import Image
import gdown # Ensure gdown is installed via requirements.txt


# --- DATABASE SETUP ---
DATABASE_URL = "sqlite:///pcb_database.db"
# Base = declarative_base() # Moved inline for Streamlit's caching
# Class definition is fine as is
# engine = create_engine(DATABASE_URL)
# Base.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)

# Use st.cache_resource for database setup to ensure it runs only once
@st.cache_resource
def get_db_session():
    Base = declarative_base() # Define Base inside the cached function
    class UploadedImage(Base):
        __tablename__ = "uploaded_images"
        id = Column(Integer, primary_key=True)
        datetime = Column(DateTime, default=datetime.now)
        image_path = Column(String)

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    st.write("--- DATABASE INITIALIZATION STARTED ---")
    try:
        import sqlalchemy
        inspector = sqlalchemy.inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            st.success("✅ Table 'uploaded_images' found in database.")
        else:
            st.warning("⚠️ Table creation attempted, but not found.")
        st.write(f"Using DB at: `{DATABASE_URL}`")
    except Exception as e:
        st.error(f"❌ DB initialization error: {e}")
    st.write("--- DATABASE INITIALIZATION FINISHED ---")
    return Session() # Return a session instance directly


# --- MODEL LOADING ---
MODEL_PATH = "models/pcb_cnn.h5"
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV" # Your confirmed model ID

@st.cache_resource
def load_and_prepare_model():
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found locally. Downloading from Google Drive...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            st.error("Please ensure the Google Drive link is publicly accessible and the file ID is correct.")
            return None

    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Deep Learning model loaded.")
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None
    return model

# --- PREDICTION FUNCTION (FIXED FOR MULTI-CLASS) ---
def predict_class(image_data, model): # Renamed to predict_class
    if model is None:
        return "Error: Model not loaded.", 0.0

    class_names = ['Burnt', 'Cu pad Damaged', 'Non-Defective', 'Rust', 'Water Damaged'] # Your original class names

    try:
        # Preprocessing must match training exactly
        img = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img) # Use renamed import
        img_array = img_array / 255.0 # Normalization
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0]).numpy() # Apply softmax for probabilities

        predicted_class_index = np.argmax(scores)
        predicted_class = class_names[predicted_class_index]
        confidence = scores[predicted_class_index] * 100

        return predicted_class, confidence

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error during prediction", 0.0

# --- STREAMLIT UI ---
def main():
    st.set_page_config(page_title="PCB Defect Classifier", layout="centered")
    st.title("🔍 PCB Board Defect Classifier")

    # Admin Login section
    with st.expander("🔐 Admin Login"):
        username = st.text_input("Username", key="admin_username")
        password = st.text_input("Password", type="password", key="admin_password")
        if st.button("Login", key="admin_login_btn"):
            if username == "PCB_Project" and password == "PCB123": # Use your original ADMIN_USERNAME/PASSWORD
                st.session_state["admin_logged_in"] = True
                st.success("✅ Admin logged in successfully!")
                st.rerun() # Rerun to update UI
            else:
                st.error("❌ Invalid credentials.")
    
    # Ensure admin_logged_in state is managed
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # Load model and handle file uploader outside of admin block
    model = load_and_prepare_model()
    uploaded_file = st.file_uploader("📤 Upload a PCB image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(image_data, caption="Uploaded Image", use_container_width=True) # Use use_container_width as suggested by logs

        if model:
            st.info("⏳ Classifying...")
            # Call the fixed predict_class function
            predicted_class, confidence = predict_class(image_data, model) 
            
            st.write(f"### 🧠 Prediction: **{predicted_class}**")
            st.write(f"**Confidence Score**: `{confidence:.2f}%`") # Display as percentage

            if predicted_class == 'Non-Defective':
                st.markdown(f"<h1 style='text-align: center; color: green;'>{predicted_class}</h1>", unsafe_allow_html=True)
            elif predicted_class.startswith("Error"):
                st.error("❌ Could not predict due to an error.")
            else: # Must be a defective class
                st.markdown(f"<h1 style='text-align: center; font-size: large; color: red;'><b>DEFECTIVE</b></h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: large;'>({predicted_class})</h1>", unsafe_allow_html=True)

            # --- Save to DB (Updated with get_db_session) ---
            try:
                session = get_db_session() # Get session from cached function
                os.makedirs("uploads", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"uploaded_image_{timestamp}_{uploaded_file.name}"
                save_path = os.path.join("uploads", image_filename)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                session.add(UploadedImage(image_path=save_path))
                session.commit()
                st.success("✅ Image and result saved to database.")
            except Exception as e:
                st.error(f"❌ DB save error: {e}")
            finally:
                session.close()
        else:
            st.error("⚠️ Model not available for prediction. Please check logs.")

    # Admin history view (if logged in)
    if st.session_state.get("admin_logged_in"):
        st.markdown("---")
        st.subheader("📊 Upload History (Admin View)")
        session = get_db_session() # Get session from cached function
        history_query = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        session.close() # Close session immediately after query

        history_display_data = []
        for item in history_query:
            history_display_data.append({
                "id": item.id,
                "Date": item.datetime.strftime("%Y-%m-%d"),
                "Time": item.datetime.strftime("%H:%M:%S"),
                "Image File": item.image_path,
                "Delete": False # Add checkbox for deletion
            })
        
        edited_df = st.data_editor(
            history_display_data,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Select to Delete",
                    help="Check to mark entries for deletion",
                    default=False
                ),
                "id": None # Hide the 'id' column from display
            },
            hide_index=True,
            key="history_table_editor"
        )
        
        # Process deletions if any checkboxes are checked
        delete_ids = [item["id"] for item in edited_df if item["Delete"]]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected Entries", key="delete_button") and delete_ids:
                session = get_db_session() # New session for deletion
                session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
                session.commit()
                session.close()
                st.success(f"Successfully deleted {len(delete_ids)} entries!")
                st.rerun()
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.rerun() # Rerun to log out and refresh UI


if __name__ == "__main__":
    main()

