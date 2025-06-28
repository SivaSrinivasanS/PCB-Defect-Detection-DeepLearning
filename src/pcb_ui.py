import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # Corrected import for load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime
<<<<<<< HEAD
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
=======
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
>>>>>>> parent of 4bdc044 (optimization)

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

<<<<<<< HEAD
# --- Model Loading (Modified Code) ---
MODEL_PATH = "models/pcb_cnn.h5"
# Google Drive File ID for pcb_cnn.h5 (YOU CONFIRMED THIS ID)
# Make sure this file is publicly shareable for anyone with the link to view/download
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV"

@st.cache_resource # Use st.cache_resource for heavy objects like models
def load_and_prepare_model():
    model = None
=======
# --- MODEL LOADING ---
MODEL_PATH = "models/pcb_cnn.h5"
GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV" # Your confirmed model ID

@st.cache_resource
def load_and_prepare_model():
>>>>>>> parent of 4bdc044 (optimization)
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(MODEL_PATH):
<<<<<<< HEAD
        st.warning(f"Model file not found at '{MODEL_PATH}'. Attempting to download from Google Drive...")
=======
        st.warning("🔄 Model not found locally. Downloading from Google Drive...")
>>>>>>> parent of 4bdc044 (optimization)
        try:
            # Use gdown to download the file directly
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            st.error("Please ensure the Google Drive link is publicly accessible and the file ID is correct.")
            return None

    try:
        model = load_model(MODEL_PATH)
<<<<<<< HEAD
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
=======
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
>>>>>>> parent of 4bdc044 (optimization)
                session.commit()
                st.success("Image uploaded and prediction performed.")
            except Exception as e:
                st.error(f"Error saving image to database: {e}")
            finally:
                session.close()
<<<<<<< HEAD
        else:
            st.error("Model not loaded. Cannot perform classification.")

if __name__ == "__main__":
    # Call database initialization once here.
    # It will run before Streamlit caches other functions if placed at module level.
    # Or, you can put this inside main() but make sure it doesn't print too much during re-runs.
    # For now, let's make it a general initialization call.
    initialize_database()
=======
                st.success(f"Successfully deleted {len(delete_ids)} entries!")
                st.rerun()
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.rerun() # Rerun to log out and refresh UI


if __name__ == "__main__":
>>>>>>> parent of 4bdc044 (optimization)
    main()

