# 🔬 **PCB Defect Detection: Industrial-Grade Computer Vision Pipeline**

Designed an automated quality inspection system to classify PCB defects with **99.22% validation accuracy**, aiming to reduce manual inspection bottlenecks and enhance quality control efficiency in electronics manufacturing.

---

## 🎯 **Engineering Challenge & Strategic Solution**

### **The Manufacturing Problem**

Manual PCB inspections are prone to:

* Human error
* Slowed throughput
* Quality control inconsistencies
  Leading to customer returns and costly downstream failures.

### **My Response**

I developed a full pipeline to convert raw PCB imagery into actionable defect classification, using deep learning to predict 5 defect types with high precision.

---

## 🚀 **Engineering Journey & Technical Highlights**

### **Career Pivot**

This project marked my transition into AI/ML engineering. I independently:

* Learned core ML concepts
* Applied practical coding knowledge
* Delivered a working deployment pipeline

### **Technical Milestones**

* **Data Engineering**: Preprocessing, augmentation pipelines
* **Modeling**: Custom CNN trained from scratch
* **App Development**: Streamlit web interface with real-time inference
* **Evaluation**: Used confusion matrix, learning curves
* **Version Control**: Managed through GitHub
* **Deployment**: Dockerized environment with Google Drive model integration

---

## ⚙️ **Custom CNN Architecture**

**Key Features:**

* Hierarchical feature learning: `Conv2D (32→64→128)`
* Efficient pooling for spatial dimension control
* ReLU activations + Dense layers for classification

**Streamlit Interface**:
Drag-and-drop image upload with real-time defect display.

**Security**:
Basic authentication (Username: `PCB_Project`, Password: `PCB123`)

**Database**:
SQLite + SQLAlchemy ORM for prediction history.

---

## 📊 **Model Performance Metrics**

| **Defect Category** | **Precision** | **Recall** | **F1-Score** | **Business Impact**          |
| ------------------- | ------------- | ---------- | ------------ | ---------------------------- |
| **Burnt PCBs**      | 99%           | 100%       | 99%          | Avoids missed defects        |
| **Cu Pad Damage**   | 100%          | 96%        | 98%          | Reduces over-scrapping       |
| **Rust**            | 100%          | 100%       | 100%         | Robust corrosion detection   |
| **Water Damage**    | 99%           | 99%        | 99%          | Improves reliability         |
| **Non-Defective**   | 99%           | 100%       | 99%          | Boosts production throughput |

* **Validation Accuracy**: `99.22%`
* **Misclassifications**: `Only 3 / 402` validation samples
* **Training Accuracy**: `100%`

---

## 📈 **Visual Analysis**

* **Learning Curves**: Smooth convergence, no overfitting
* **Confusion Matrix**: Strong diagonal dominance (indicating confident predictions)

---

## ▶️ **Live Demo & Screenshots**

**🔗 Streamlit App**:
[Launch Live Demo](https://pcb-defect-detection-deeplearning-app.streamlit.app/)

(Optional)

* Add GIF demo: `assets/live_demo_gif.gif`
* Screenshot: `assets/ui_screenshot.png`

---

## 🧪 **How to Run Locally**

### **1. Clone & Setup**

```bash
git clone https://github.com/SivaSrinivasanS/PCB-Defect-Detection-DeepLearning.git
cd PCB-Defect-Detection-DeepLearning
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### **2. Download Model**

**Pretrained Model (.h5)**:
[Download pcb\_cnn.h5](https://drive.google.com/file/d/1rLOP-q2c_cw0UraOXIUNeI5eKFIX3uFV/view?usp=drive_link)

Place in:
`models/pcb_cnn.h5`

### **3. Launch Application**

```bash
streamlit run src/pcb_ui.py
```

---

## 🛠️ **Future Enhancements**

* **YOLOv8 Integration** for bounding-box defect detection
* **EfficientNet/ResNet** for transfer learning comparison
* **Edge Deployment** for <100ms inference latency
* **Active Learning Loop** using human-in-the-loop corrections
* **PostgreSQL Upgrade** for multi-user database access
* **CI/CD** pipelines using GitHub Actions
* **Cloud Containerization** via Docker/Kubernetes

---

## 📁 **Repository Structure**

```
PCB-Defect-Detection-DeepLearning/
├── .github/                   # CI/CD workflows
│   └── workflows/
├── assets/                    # Demo images and visualizations
│   ├── accuracy_loss_plot.png
│   ├── confusion_matrix_heatmap.png
│   ├── ui_screenshot.png
│   └── live_demo_gif.gif
├── models/                    # Trained model(s)
│   └── pcb_cnn.h5
├── src/                       # Source code
│   ├── train_model.ipynb
│   └── pcb_ui.py
├── data/                      # (Optional) Dataset folders
│   └── augmented_dataset/
├── requirements.txt           # Pinned dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .gitignore                 # Ignore list
```

---

## 👤 **Author**

**Siva Srinivasan S**
[LinkedIn Profile](https://www.linkedin.com/in/sivasrinivasans/)

---

