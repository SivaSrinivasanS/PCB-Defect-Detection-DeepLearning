🔬 PCB Defect Detection: Industrial-Grade Computer Vision Pipeline
Architected an automated quality inspection system that helps revolutionize PCB manufacturing workflows, achieving 99.22% validation accuracy while aiming to significantly reduce manual inspection bottlenecks and improve overall quality control.

🎯 Engineering Challenge & Strategic Solution
The Manufacturing Crisis:
Electronics production lines often face inefficiencies from manual PCB inspection processes. These are frequently plagued by human error rates, inspection throughput bottlenecks that slow down production, and quality control inconsistencies. Such issues can lead to costly downstream failures and customer returns.

My Engineering Response:
I conceived and delivered an end-to-end computer vision solution that helps transform raw PCB imagery into actionable manufacturing intelligence. This system autonomously classifies 5 critical defect categories with high precision, enabling more informed production decisions and contributing to improved quality control in manufacturing.

🚀 Personal Engineering Journey & Technical Leadership
Career Transition & Skill Development:
This project represents a pivotal milestone in my strategic pivot into AI/ML engineering. Starting with foundational coding knowledge, I focused on learning and applying various aspects of building a complex AI system.

Independent Technical Achievements:
Data Engineering: I developed robust preprocessing and augmentation pipelines, transforming raw manufacturing imagery into datasets suitable for machine learning.

Model Development: I designed and trained a Convolutional Neural Network (CNN) from scratch to classify PCB defects, achieving strong performance for this specific task.

Application Development: I built a functional web application with Streamlit, including basic database persistence and user authentication for real-time inference capabilities.

Comprehensive Validation: I implemented thorough evaluation frameworks, including confusion matrices and learning curve analysis, to understand model performance.

Version Control: I learned to use Git for managing code versions and organizing the project effectively, reflecting a foundation in professional development practices.

Technical Growth Trajectory:
This hands-on experience has been crucial for building my practical skills across deep learning model development, MLOps concepts, and application deployment. It demonstrates my ability to rapidly absorb complex technical domains and work towards delivering business-relevant solutions.

⚙️ Advanced Technical Architecture & Innovation
Custom CNN Design Philosophy:
I engineered a hierarchical feature extraction network specifically designed for recognizing PCB defect patterns. This domain-specific architecture aims to capture the unique geometric patterns and electrical trace anomalies critical for electronics manufacturing quality control.

Architectural Innovation Highlights:
Progressive Filter Expansion (32→64→128): This enables learning multi-scale features from basic edge detection to more complex defect pattern recognition.

Spatial Dimension Management: Strategic MaxPooling placement helps preserve critical defect boundary information while contributing to computational efficiency.

Activation Strategy: ReLU functions are used with carefully configured dense layers to aid in stable training and classification.

System Design Overview:
Web Application Layer: The Streamlit-based interface offers intuitive drag-and-drop functionality for image upload, real-time confidence scoring, and responsive design principles.

Data Persistence Architecture: A SQLite backend with SQLAlchemy ORM provides audit trails, prediction history, and basic admin workflows.

Security Framework: A role-based authentication system (Username: PCB_Project / Password: PCB123) is included for administrative access.

Deployment Approach: The architecture is designed to be containerization-ready, facilitating environment management and dependency isolation for future deployment.

📊 Quantified Performance Engineering
Model Performance:
Training Convergence: Achieved 100% training accuracy with 99.22% validation performance, demonstrating effective learning and generalization.

Learning Efficiency: Rapid stabilization by epoch 5, indicating efficient hyperparameter choices and architecture design.

Generalization Robustness: Minimal validation loss (0.0138) confirms strong real-world deployment viability.

| Defect Category | Precision | Recall | F1-Score | Business Impact |
| :---------------- | :-------- | :----- | :------- | :------------------------------------------------------------------ |
| **Burnt PCBs** | 99% | 100% | 99% | Helps prevent defective units from shipping by minimizing false negatives. |
| **Cu Pad Damage** | 100% | 96% | 98% | High precision helps avoid scrapping good units. |
| **Rust Detection** | 100% | 100% | 100% | Strong identification of corrosion patterns. |
| **Water Damage** | 99% | 99% | 99% | Helps prevent reliability failures due to moisture. |
| **Non-Defective** | 99% | 100% | 99% | Supports optimal throughput with very few false positives. |

Confusion Matrix Analysis:
The analysis of the confusion matrix reveals excellent classification. With only 3 total misclassifications across 402 validation samples, it demonstrates a high rate of correct predictions.

📈 Visual Analysis
Learning Curves:

Analysis: Both training and validation accuracy and loss curves show healthy progress, converging smoothly without signs of severe overfitting. This indicates reliable model training.

Confusion Matrix:

Analysis: The visual representation of the confusion matrix confirms the model's high accuracy, with most predictions falling on the correct diagonal.

▶️ Live Demo & Screenshots
Experience the application live and upload your own PCB images!

Live Streamlit App: [INSERT YOUR STREAMLIT CLOUD URL HERE]


(Screenshot of the main application interface, showing a prediction example.)


(Optional: A short GIF demonstrating the image upload and prediction process.)

🏃 How to Run Locally
To set up and run this project on your local machine:

Local Development Setup:
# Repository cloning and environment preparation
git clone https://github.com/SivaSrinivasanS/PCB-Defect-Detection-DeepLearning.git
cd PCB-Defect-Detection-DeepLearning

# Virtual environment creation and activation
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Dependency installation
pip install -r requirements.txt

Dataset Acquisition and Placement:
The full dataset is NOT included in this repository due to its size, nor are any sample images.

To test the Streamlit app or avoid retraining, download the pretrained model:
🔗 **Pretrained Model (.h5)**: [Download pcb_cnn.h5 from Google Drive](https://drive.google.com/file/d/1C3n4XFUsD6FiqcS79BA1pThGTcfEV0IG/view?usp=sharing)

Extract its contents and place it into a new folder named data/augmented_dataset/ within your cloned project directory. The final path should be your-repo-root/data/augmented_dataset/.

Alternatively, to test the Streamlit application without full training data, you can download any relevant PCB images (both defective and non-defective) from public sources. Ensure they are clear and representative of defect types.

Model Training & Deployment:
# Launch Jupyter for model training
jupyter notebook
# Execute all cells in: src/train_model.ipynb

# Start the Streamlit web application
streamlit run src/pcb_ui.py

Admin Access: Username: PCB_Project / Password: PCB123

🛠️ Future Technical Roadmap & Innovation Pipeline
Advanced Computer Vision Extensions:
Object Detection Integration: Explore implementing YOLOv8 for precise defect localization with bounding box generation.

Model Enhancement: Investigate using other deep learning architectures or pre-trained models (e.g., EfficientNet, ResNet) for performance comparison.

Real-Time Processing: Explore edge computing deployment for manufacturing floor integration with sub-100ms inference latency.

MLOps & Infrastructure Evolution:
Automated Retraining Pipeline: Develop capabilities for data drift detection and continuous learning.

Docker Containerization: Implement Docker for production environment consistency and Kubernetes orchestration readiness.

Cloud Database Migration: Explore PostgreSQL or similar for multi-user scalability and enterprise data management.

Feedback Loop Implementation: Create an active learning system incorporating user corrections for continuous model improvement.

📁 Professional Repository Architecture
PCB-Defect-Detection-DeepLearning/
├── .github/                     # CI/CD workflows and automation (e.g., for Streamlit Cloud deployment)
│   └── workflows/               # GitHub Actions for testing and deployment
├── .gitignore                   # Comprehensive exclusion patterns for Git
├── README.md                    # Executive technical documentation and project overview
├── LICENSE                      # MIT open-source licensing details
├── requirements.txt             # Pinned Python dependency management
├── src/                         # All primary source code modules
│   ├── train_model.ipynb        # Jupyter notebook for model development and evaluation
│   └── pcb_ui.py                # Production-ready Streamlit web application
├── models/                      # Trained model artifacts and metadata
│   └── pcb_cnn.h5               # Primary production model (may require Git LFS if large)
└── assets/                      # Visual documentation and demonstrations for README.md
    ├── accuracy_loss_plot.png   # Training performance visualizations (learning curves)
    ├── confusion_matrix_heatmap.png # Classification analysis (confusion matrix)
    ├── ui_screenshot.png        # Application interface documentation (screenshot)
    └── live_demo_gif.gif        # Interactive demonstration (optional GIF)

Author: Siva Srinivasan S
LinkedIn Profile: https://www.linkedin.com/in/sivasrinivasans/

This project showcases comprehensive ML engineering excellence—from data preprocessing and custom model architecture to comprehensive evaluation and web deployment—delivering quantifiable business value in manufacturing quality control while demonstrating rapid technical mastery and end-to-end system ownership.