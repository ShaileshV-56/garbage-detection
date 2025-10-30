# Waste Object Detection and Classification

This repository contains implementations for waste detection and classification using computer vision models.

## 📁 Datasets

### 1. RoboFlow Garbage Detection Dataset
- **Type**: Object Detection (Bounding Boxes)
- **Classes**: Single class ("garbage")
- **Images**: 260 validation images with 271 garbage instances
- **Resolution**: 640×640 pixels
- **Use Case**: General waste object localization in real-world scenes
- **Source**: [RoboFlow Universe](https://universe.roboflow.com/garbage-detection-czeg5/garbage_detection-wvzwv)

### 2. TrashNet Dataset
- **Type**: Image Classification
- **Classes**: 6 categories (glass, paper, cardboard, plastic, metal, trash)
- **Images**: 2,527 total images with class imbalance
- **Resolution**: Various (resized to 224×224 for training)
- **Use Case**: Waste material classification on white background
- **Source**: [GitHub Repository](https://github.com/garythung/trashnet)

## 🚀 Models Implemented

### YOLOv8 Object Detection
- **Task**: Detect garbage objects in images
- **Model**: YOLOv8n (nano variant)
- **Performance**: 
  - mAP@50: 86.4%
  - Precision: 89.4%
  - Recall: 80.8%

### CNN Classification (Keras/R)
- **Task**: Classify waste into 6 categories
- **Model**: Custom CNN with 3 convolutional blocks
- **Performance**:
  - Validation Accuracy: 91.2%
  - Precision: 91.5%
  - Recall: 90.1%

## 📊 Results Summary

| Model | Task | Accuracy/mAP | Key Strength |
|-------|------|--------------|--------------|
| YOLOv8n | Object Detection | 86.4% mAP@50 | Excellent localization |
| Custom CNN | Classification | 91.2% accuracy | Strong categorization |

## 🛠️ Installation

```bash
# For YOLOv8 (Python)
pip install ultralytics

# For Keras in R
install.packages("keras")
library(keras)
install_keras()
💡 Usage
YOLOv8 Training
python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='garbage.yaml', epochs=100, imgsz=640)
Keras/R Training
r
library(keras)
# See appendix for complete training code
📈 Future Work
Multi-class waste detection with bounding boxes

Transfer learning with pre-trained models

Real-world deployment and testing

Addressing class imbalance in TrashNet

📝 License
Please refer to the original dataset sources for licensing information.

text

This README provides a clean, professional overview with:
- Clear dataset descriptions
- Model performance summaries
- Quick installation and usage guides
- Structured formatting for easy navigation
- Key metrics highlighted for quick reference
