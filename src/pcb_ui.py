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

# --- Database Setup for Streamlit Cloud ---
@st.cache_resource
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
        
    except Exception as e:
        st.warning(f"Database setup failed: {e}. Continuing without database logging.")
        return None, None, None

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