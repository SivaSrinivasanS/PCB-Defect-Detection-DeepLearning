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
import gdown # Used for model download fallback, though Dockerfile is primary

# --- Configuration Constants ---
HISTORY_KEY = "uploaded_images"
ADMIN_USERNAME = "TANSAM_TIDEL" # Your specified admin username
ADMIN_PASSWORD = "TANSAM123" # Your specified admin password

# --- Database Setup ---
Base = declarative_base()

# Determine the project root directory from the current script's location
# This script (pcb_ui.py) is in the 'src' folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to reach the main project root (e.g., 'PCB-Defect-Detection-DeepLearning')
project_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Define the path for the SQLite database file in the project root
db_path = os.path.join(project_root_dir, 'pcb_database.db')

# Create the SQLAlchemy engine. echo=False to reduce log verbosity in production.
engine = create_engine(f'sqlite:///{db_path}', echo=False)

# Database model for uploaded images
class UploadedImage(Base):
    __tablename__ = "uploaded_images" # This line is now clean of U+00A0
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String) # Stores the full path to the saved image file

# Use st.cache_resource to ensure database initialization runs only once across the app's lifecycle.
# This prevents re-creating tables on every Streamlit rerun.
@st.cache_resource
def initialize_database():
    # Print statements will go to Streamlit Cloud logs (accessible from 'Manage app' -> 'Logs')
    print(f"\n--- DATABASE INITIALIZATION STARTED ---")
    print(f"Checking/Creating database at: {db_path}")

    try:
        # Create tables (if they don't exist). This is idempotent and safe to call multiple times.
        Base.metadata.create_all(engine)
        print(f"Database tables created/checked successfully for engine at {db_path}.")

        # Verify if the table exists using an inspector (for debugging purposes)
        inspector = inspect(engine)
        if 'uploaded_images' in inspector.get_table_names():
            print("CONFIRMATION: Table 'uploaded_images' found in database.")
        else:
            print("WARNING: Table 'uploaded_images' NOT found in database after create_all!")
            st.error("Failed to create 'uploaded_images' table. Check permissions or database path.")
            return False # Indicate initialization failure to stop the app

        # Test database connection with a simple query
        test_session = sessionmaker(bind=engine)()
        test_session.execute(text("SELECT 1")) # Use text() for literal SQL to avoid warnings
        test_session.close()
        print("Database connection successfully tested.")

    except Exception as e:
        print(f"CRITICAL ERROR during database initialization: {e}")
        st.error(f"A critical error occurred during database setup: {e}. "
                 "Please ensure the database file is not locked by another process "
                 "and that you have write permissions to the project directory.")
        return False # Indicate failure

    print(f"--- DATABASE INITIALIZATION FINISHED ---")
    return True # Indicate successful initialization

# Call the initialization function when the app starts. If it fails, stop the Streamlit app.
if not initialize_database():
    st.stop()

# Create the Session class after the engine is initialized and tables are potentially created
Session = sessionmaker(bind=engine)


# --- Model Loading and Prediction Function ---

# Use st.cache_resource to load the model only once, making the app faster.
@st.cache_resource
def load_and_prepare_model():
    model_path = os.path.join(project_root_dir, 'models', 'pcb_cnn.h5')
    model_dir = os.path.dirname(MODEL_PATH)

    # Ensure the 'models' directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # If model file is not found locally, attempt to download it from Google Drive.
    # This serves as a fallback for local testing, as the Dockerfile will handle
    # the primary download for deployment.
    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found locally. Attempting to download from Google Drive...")
        GOOGLE_DRIVE_FILE_ID = "1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV" # Your confirmed model ID
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            st.error("Please ensure the Google Drive link is publicly accessible and the file ID is correct.")
            return None # Return None if download fails

    try:
        # Load the TensorFlow Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}") # Log to console
        return model
    except Exception as e:
        st.error(f"Error loading model from '{MODEL_PATH}': {e}")
        return None

# Load the model globally when the app starts using the cached function
classifier_model = load_and_prepare_model()

# Prediction function with multi-class logic and enhanced output
def predict_class(image_data_bytes, model): # Takes image data as bytes and the model object
    if model is None:
        return "Error: Prediction model not available.", 0.0 # Return error message and 0 confidence

    img_h, img_w = 224, 224
    # Ensure class names are in the exact order as trained by your model (from ImageDataGenerator)
    # This is crucial for correct mapping of prediction indices to human-readable labels.
    class_names = ['Burnt', 'Cu pad Damaged', 'Non-Defective', 'Rust', 'Water Damaged']

    try:
        # Preprocess the input image: Open from bytes, convert to RGB, resize
        img = Image.open(io.BytesIO(image_data_bytes)).convert("RGB").resize((img_h, img_w))
        img_array = keras_image.img_to_array(img) # Convert to NumPy array
        img_array = img_array / 255.0 # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)

        predictions = model.predict(img_array, verbose=0) # Use verbose=0 to suppress Keras prediction output to console
        scores = tf.nn.softmax(predictions[0]).numpy() # Apply softmax to get probabilities and convert to NumPy array
        
        predicted_class_index = np.argmax(scores) # Get the index of the class with the highest probability
        predicted_class = class_names[predicted_class_index] # Map the predicted index to its class name
        confidence = scores[predicted_class_index] * 100 # Convert confidence score to percentage

        return predicted_class, confidence

    except Exception as e:
        st.error(f"An error occurred during prediction preprocessing or model inference: {e}")
        # Return a clear error message and 0 confidence if prediction fails
        return "Error during prediction", 0.0

# --- Streamlit Main Application Logic ---
def main():
    # Configure the Streamlit page layout and title
    st.set_page_config(layout="centered", page_title="PCB Board Predictor")
    st.title("PCB Board Class Predictor")
    st.markdown("Upload an image of a PCB board to predict its class.")

    # Initialize Streamlit session state variables if they don't exist
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False
    if "redirect" not in st.session_state:
        st.session_state["redirect"] = False

    admin_logged_in = st.session_state.get("admin_logged_in", False)

    # Logic for Admin Login vs. Regular User UI
    if not admin_logged_in:
        # Button to initiate admin login process
        if st.button("Admin Login", key="admin_login_button"):
            st.session_state["redirect"] = True
        
        # Display login fields if redirect state is true
        if st.session_state.get("redirect", False):
            st.markdown("---")
            st.subheader("Admin Login")
            username = st.text_input("Username", key="admin_username")
            password = st.text_input("Password", type="password", key="admin_password")
            if st.button("Login", key="login_button_submit"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["redirect"] = False # Reset redirect state
                    st.success("Logged in successfully!")
                    st.rerun() # Rerun the app to update UI based on login status
                else:
                    st.error("Invalid username or password")
        else: # Regular user view (not logged in admin)
            file_uploaded = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
            if file_uploaded is not None:
                # Read image data as bytes
                image_data_bytes = file_uploaded.read()

                # Check if the model was loaded successfully before attempting prediction
                if classifier_model is None:
                    st.warning("Cannot classify: Deep Learning model not loaded. "
                               "Please check the console for errors during model loading, "
                               "and ensure 'train_model.ipynb' was run successfully to save the model.")
                    return # Exit the function early if model is not available

                try:
                    # Display the uploaded image
                    st.image(image_data_bytes, caption='Uploaded Image', use_container_width=True)
                    
                    with st.spinner('Classifying...'): # Display a spinner during prediction
                        predicted_class, confidence = predict_class(image_data_bytes, classifier_model)
                    
                    # Display results clearly
                    if predicted_class.startswith("Error"):
                        st.error(f"Prediction Failed: {predicted_class}")
                    else:
                        st.markdown(f"**Predicted Class:** <span style='font-size: large;'>{predicted_class}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** <span style='font-size: large;'>{confidence:.2f}%</span>", unsafe_allow_html=True)

                        if predicted_class == 'Non-Defective':
                            st.markdown(f"<h1 style='text-align: center; color: green;'>{predicted_class}</h1>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h1 style='text-align: center; font-size: large; color: red;'><b>DEFECTIVE</b></h1>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='text-align: center; font-size: large;'>({predicted_class})</h1>", unsafe_allow_html=True)
                    
                    # --- Save to DB ---
                    session = Session() # Create a new session for database operation
                    # Ensure 'uploads' directory exists relative to the project root
                    uploads_dir = os.path.join(project_root_dir, "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"uploaded_image_{timestamp}_{file_uploaded.name}"
                    save_path = os.path.join(uploads_dir, image_filename) # Save full path to the file
                    
                    # Save the image file to the 'uploads' directory
                    with open(save_path, "wb") as f:
                        f.write(image_data_bytes)
                        
                    # Record the image path and datetime in the database
                    new_image = UploadedImage(datetime=datetime.now(), image_path=save_path)
                    session.add(new_image)
                    session.commit()
                    session.close() # Always close the session after use
                    st.success("✅ Image and result saved to database.")

                except Exception as e:
                    st.error("An unexpected error occurred during image processing or prediction.")
                    st.write(f"Error details: {e}")

    # Admin history view (if logged in)
    if admin_logged_in:
        st.markdown("---")
        st.subheader("Upload History")
        session = Session() # Create a new session for querying history
        history_query = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        session.close() # Close session immediately after query

        history_display_data = []
        for item in history_query:
            history_display_data.append({
                "id": item.id,
                "Date": item.datetime.strftime("%Y-%m-%d"),
                "Time": item.datetime.strftime("%H:%M:%S"),
                "Image File": item.image_path, # Display the full path
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
                "id": None # Hide the 'id' column from display
            },
            hide_index=True,
            key="history_table_editor"
        )

        delete_ids = [item["id"] for item in edited_df if item["Delete"]]

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected Entries", key="delete_button") and delete_ids:
                session = Session() # New session for deletion operations
                # Delete image files from 'uploads' directory first
                for img_id in delete_ids:
                    img_record = session.query(UploadedImage).filter_by(id=img_id).first()
                    if img_record and img_record.image_path and os.path.exists(img_record.image_path):
                        try:
                            os.remove(img_record.image_path)
                            print(f"Deleted image file from disk: {img_record.image_path}")
                        except Exception as file_e:
                            print(f"Error deleting file {img_record.image_path} from disk: {file_e}")
                            st.error(f"Could not delete file {os.path.basename(img_record.image_path)}. Please check permissions.")
                
                # Then delete records from the database
                session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
                session.commit()
                session.close()
                st.success(f"Successfully deleted {len(delete_ids)} entries and associated files!")
                st.rerun() # Rerun to refresh the history table
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.session_state["redirect"] = False
                st.success("Logged out successfully!")
                st.rerun()
