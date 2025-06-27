import os
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Configuration Constants ---
HISTORY_KEY = "uploaded_images"
ADMIN_USERNAME = "PCB_Project"
ADMIN_PASSWORD = "PCB123"

# --- Database Setup ---
Base = declarative_base()

# Determine the project root directory from the current script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Define the path for the SQLite database file in the project root
db_path = os.path.join(project_root_dir, 'pcb_database.db')

engine = create_engine(f'sqlite:///{db_path}', echo=True)

# Database model for uploaded images
class UploadedImage(Base):
    __tablename__ = 'uploaded_images'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String)

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
            print("WARNING: Table 'uploaded_images' NOT found in database after create_all!")
            st.error("Failed to create 'uploaded_images' table. Check permissions or database path.")
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

# Initialize the database. If it fails, stop the Streamlit app.
if not initialize_database():
    st.stop()

# Create the Session class after the engine is initialized
Session = sessionmaker(bind=engine)

# --- Model Loading and Prediction Function ---
@st.cache_resource
def load_model():
    model_path = os.path.join(project_root_dir, 'models', 'pcb_cnn.h5')
    
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. "
                 "Please ensure 'train_model.ipynb' has been run successfully "
                 "to save the model in the 'models' folder.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

# Load the model globally when the app starts
classifier_model = load_model()

# Prediction function with enhanced output
def predict_class(image):
    if classifier_model is None:
        return "Error: Prediction model not available.", 0.0 # Return error message and 0 confidence

    img_h, img_w = 224, 224

    try:
        test_image = image.resize((img_h, img_w))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        class_names = ['Burnt', 'Cu pad Damaged', 'Non-Defective', 'Rust', 'Water Damaged']

        predictions = classifier_model.predict(test_image)
        scores = tf.nn.softmax(predictions[0]).numpy() # Get probabilities and convert to NumPy array
        
        predicted_class_index = np.argmax(scores)
        predicted_class = class_names[predicted_class_index]
        confidence = scores[predicted_class_index] * 100 # Convert to percentage

        return predicted_class, confidence

    except Exception as e:
        st.error(f"An error occurred during prediction preprocessing or model inference: {e}")
        return "Error during prediction", 0.0 # Return error message and 0 confidence

# --- Streamlit Main Application Logic ---
def main():
    st.set_page_config(layout="centered", page_title="PCB Board Predictor")
    st.title("PCB Board Class Predictor")
    st.markdown("Upload an image of a PCB board to predict its class.")

    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False
    if "redirect" not in st.session_state:
        st.session_state["redirect"] = False

    admin_logged_in = st.session_state.get("admin_logged_in", False)

    if not admin_logged_in:
        if st.button("Admin Login", key="admin_login_button"):
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
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        else:
            file_uploaded = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
            if file_uploaded is not None:
                if classifier_model is None:
                    st.warning("Cannot classify: Deep Learning model not loaded. "
                               "Please check the console for errors during model loading, "
                               "and ensure 'train_model.ipynb' was run to save the model.")
                    return

                try:
                    image = Image.open(file_uploaded)
                    st.image(image, caption='Uploaded Image', use_container_width=True) # Corrected parameter
                    
                    # Create a placeholder for "Classifying..." message
                    status_placeholder = st.empty()
                    status_placeholder.write("Classifying...")
                    
                    with st.spinner('Predicting...'):
                        predicted_class, confidence = predict_class(image)
                    
                    # Clear the "Classifying..." message after prediction
                    status_placeholder.empty()

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
                    
                    session = Session()
                    new_image = UploadedImage(datetime=datetime.now(), image_path=file_uploaded.name)
                    session.add(new_image)
                    session.commit()
                    session.close()
                except Exception as e:
                    st.error("An error occurred during image processing or result display.")
                    st.write(f"Error details: {e}")

    if admin_logged_in:
        st.markdown("---")
        st.subheader("Upload History")
        session = Session()
        history_query = session.query(UploadedImage).order_by(UploadedImage.datetime.desc()).all()
        
        history_display_data = []
        for item in history_query:
            history_display_data.append({
                "id": item.id,
                "Date": item.datetime.strftime("%Y-%m-%d"),
                "Time": item.datetime.strftime("%H:%M:%S"),
                "Image File": item.image_path,
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
                "id": None
            },
            hide_index=True,
            key="history_table_editor"
        )

        delete_ids = [item["id"] for item in edited_df if item["Delete"]]

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected Entries", key="delete_button") and delete_ids:
                session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
                session.commit()
                session.close()
                st.success(f"Successfully deleted {len(delete_ids)} selected entries!")
                st.rerun()
        with col2:
            if st.button("Logout", key="logout_button"):
                st.session_state["admin_logged_in"] = False
                st.session_state["redirect"] = False
                st.success("Logged out successfully!")
                st.rerun()
        
        session.close()

if __name__ == "__main__":
    main()
