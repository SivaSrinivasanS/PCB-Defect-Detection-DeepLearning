import streamlit as st
import tensorflow as tf
<<<<<<< HEAD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as keras_image
=======
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image # Renamed to avoid conflict with PIL Image
>>>>>>> parent of 057d6f0 (Revert "optimization")
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import io
from PIL import Image
<<<<<<< HEAD
import gdown
import json
import h5py
import tempfile
import time

st.set_page_config(
    page_title="PCB Defect Classifier", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Configuration Constants ---
ADMIN_USERNAME = "PCB_project"
ADMIN_PASSWORD = "PCB123"
MODEL_FILE_ID = "1C3n4XFUsD6FiqcS79BA1pThGTcfEV0IG"
=======
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
>>>>>>> parent of 057d6f0 (Revert "optimization")

# --- Database Setup for Streamlit Cloud ---
@st.cache_resource
<<<<<<< HEAD
def setup_database():
    """Initialize database with proper error handling for Streamlit Cloud"""
    try:
        # Use temporary directory for Streamlit Cloud
        temp_dir = tempfile.gettempdir()
        db_path = os.path.join(temp_dir, 'pcb_database.db')
        
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        Base = declarative_base()
        
        class UploadedImage(Base):
            __tablename__ = "uploaded_images"
            id = Column(Integer, primary_key=True)
            datetime = Column(DateTime)
            image_path = Column(String)
        
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        
        return engine, Session, UploadedImage
        
=======
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
>>>>>>> parent of 057d6f0 (Revert "optimization")
    except Exception as e:
        st.warning(f"Database setup failed: {e}. Continuing without database logging.")
        return None, None, None

<<<<<<< HEAD
# Setup database
db_engine, db_session_class, UploadedImage = setup_database()

# --- Class Mapping ---
CLASS_MAP = {
    "0": "Burnt",
    "1": "Cu pad Damaged",
    "2": "Non-Defective", 
    "3": "Rust",
    "4": "Water Damaged"
}

# --- Streamlit Cloud Compatible Model Loading ---
@st.cache_resource(show_spinner=False)
def download_and_load_model():
    """Download and load model with Streamlit Cloud optimizations"""
    
    # Create temporary directory for model
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'pcb_cnn.h5')
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text('🔄 Downloading AI model... (This may take 2-3 minutes)')
        progress_bar.progress(10)
        
        # Download with gdown using direct link method
        download_url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        
        # Try gdown download with better error handling
        try:
            progress_bar.progress(20)
            gdown.download(id=MODEL_FILE_ID, output=model_path, quiet=True, fuzzy=True)
            progress_bar.progress(60)
            
        except Exception as e:
            status_text.text(f'⚠️ Standard download failed: {e}. Trying alternative method...')
            # Alternative download method
            import urllib.request
            urllib.request.urlretrieve(download_url, model_path)
            progress_bar.progress(60)
        
        # Verify file was downloaded
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:  # Less than 1MB means failed
            raise Exception("Model file download incomplete or corrupted")
        
        status_text.text('✅ Model downloaded! Loading into memory...')
        progress_bar.progress(70)
        
        # Load model with multiple fallback methods
        model = None
        
        # Method 1: Standard Keras load
        try:
            model = tf.keras.models.load_model(model_path)
            status_text.text('✅ Model loaded successfully!')
            progress_bar.progress(100)
            
        except Exception as e1:
            status_text.text('🔄 Trying alternative loading method...')
            progress_bar.progress(80)
            
            # Method 2: Manual loading with h5py
            try:
                with h5py.File(model_path, 'r') as f:
                    # Get model config
                    model_config = f.attrs.get('model_config')
                    if model_config is None:
                        raise Exception("No model configuration found in .h5 file")
                    
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    
                    # Create model from JSON and load weights
                    model = model_from_json(model_config)
                    model.load_weights(model_path)
                    
                    status_text.text('✅ Model loaded using alternative method!')
                    progress_bar.progress(100)
                    
            except Exception as e2:
                raise Exception(f"All loading methods failed. Error 1: {e1}, Error 2: {e2}")
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return model
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Failed to download/load model: {e}")
        st.info("💡 **Troubleshooting Tips:**\n"
                "- Refresh the page and try again\n"
                "- Check your internet connection\n"
                "- The model file is ~134MB and may take time to download")
        return None

# --- Prediction Function ---
def predict_defect(image_data, model):
    """Make prediction on uploaded image"""
    if model is None:
        return "Model not available", 0.0
    
    try:
        # Process image
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        probabilities = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        label = CLASS_MAP.get(str(predicted_index), f"Class {predicted_index}")
        
        return label, confidence, probabilities
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error during prediction", 0.0, None
=======
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
>>>>>>> parent of 057d6f0 (Revert "optimization")

# --- Save to Database (with error handling) ---
def save_to_database(filename, prediction, confidence):
    """Save prediction to database if available"""
    if db_session_class and UploadedImage:
        try:
            session = db_session_class()
            new_record = UploadedImage(
                datetime=datetime.now(),
                image_path=f"{filename} - {prediction} ({confidence:.1%})"
            )
            session.add(new_record)
            session.commit()
            session.close()
            return True
        except Exception as e:
            st.warning(f"Database logging failed: {e}")
            return False
    return False

# --- Main Application ---
def main():
<<<<<<< HEAD
    # Header
    st.title("🔬 PCB Defect Detection System")
    st.markdown("**Industrial-Grade Computer Vision Pipeline** | Upload a PCB image to detect defects using AI")
    
    # Load model
    with st.spinner("🚀 Initializing AI model..."):
        classifier_model = download_and_load_model()
    
    if classifier_model is None:
        st.error("❌ **Cannot proceed without the AI model.** Please refresh the page to retry.")
        st.stop()
        
    st.success("🎯 **AI Model Ready!** Upload your PCB image below.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "**Choose a PCB image for analysis**", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a clear, well-lit image of the PCB for accurate defect detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 **Uploaded PCB Image**")
            image_data = uploaded_file.read()
            st.image(image_data, use_column_width=True, caption=f"File: {uploaded_file.name}")
        
        with col2:
            st.subheader("🎯 **AI Analysis Results**")
            
            with st.spinner("🔍 Analyzing PCB defects..."):
                prediction, confidence, probabilities = predict_defect(image_data, classifier_model)
            
            if "Error" in prediction:
                st.error(f"❌ {prediction}")
            else:
                # Main result
                if "Non-Defective" in prediction:
                    st.success(f"✅ **Result:** {prediction}")
                    st.balloons()  # Celebration for good PCBs!
                else:
                    st.warning(f"⚠️ **Defect Detected:** {prediction}")
                
                # Confidence metrics
                st.metric("🎯 Confidence Level", f"{confidence:.1%}")
                
                # Confidence bar
                if confidence > 0.85:
                    st.success("🔥 **High Confidence** - Reliable prediction")
                elif confidence > 0.70:
                    st.info("👍 **Good Confidence** - Trustworthy result")
                else:
                    st.warning("🤔 **Moderate Confidence** - Consider retaking image")
                
                # Detailed probabilities
                if probabilities is not None:
                    with st.expander("📊 **Detailed Classification Scores**"):
                        for idx, (class_id, class_name) in enumerate(CLASS_MAP.items()):
                            prob = probabilities[int(class_id)]
                            st.write(f"**{class_name}:** {prob:.1%}")
                            st.progress(prob)
                
                # Save to database
                save_to_database(uploaded_file.name, prediction, confidence)
    
    # Sidebar Information
    with st.sidebar:
        st.header("ℹ️ **About This System**")
        st.markdown("""
        This AI system can detect **5 types of PCB defects**:
        """)
        
        for class_id, class_name in CLASS_MAP.items():
            if "Non-Defective" in class_name:
                st.success(f"✅ {class_name}")
            else:
                st.error(f"⚠️ {class_name}")
        
        st.header("📊 **Performance Stats**")
        st.metric("Model Accuracy", "99.22%")
        st.metric("Validation Samples", "402")
        st.metric("Training Epochs", "10")
        
        st.header("💡 **Tips for Best Results**")
        st.markdown("""
        - **Good lighting** - Avoid shadows
        - **Clear focus** - No blurry images  
        - **Full PCB visible** - Complete board view
        - **Stable camera** - Minimize motion blur
        - **High resolution** - Better detail capture
        """)
        
        st.header("🔐 **Admin Panel**")
        with st.expander("Admin Login"):
            username = st.text_input("Username", key="admin_user")
            password = st.text_input("Password", type="password", key="admin_pass")
            
            if st.button("🔓 Admin Access"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.success("✅ **Admin Access Granted**")
                    st.info("📈 **System Status:** Operational\n🤖 **Model:** PCB CNN v1.0\n💾 **Database:** Connected")
                else:
                    st.error("❌ Invalid credentials")
        
        st.markdown("---")
        st.markdown("**Developed by:** Siva Srinivasan S")
        st.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/sivasrinivasans/)")

if __name__ == "__main__":
    main()
=======
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

>>>>>>> parent of 057d6f0 (Revert "optimization")
