# Garbage Detection on Streets using YOLOv8

Real-time garbage detection system for street-level waste monitoring using YOLOv8 object detection.

## 📁 Dataset

### RoboFlow Street Garbage Detection
- **Type**:        Object Detection (Bounding Boxes)
- **Classes**:     Single class ("garbage")
- **Images**:      260 validation images with 271 garbage instances
- **Resolution**:  640×640 pixels
- **Characteristics**: Real-world street scenes with multiple garbage objects
- **Source**:      [RoboFlow Universe](https://universe.roboflow.com/garbage-detection-czeg5/garbage_detection-wvzwv)

## 🚀 Model

### YOLOv8 Nano (Object Detection)
- **Architecture**: YOLOv8n - optimized for speed and accuracy
- **Training**:     100 epochs with transfer learning from COCO weights
- **Input Size**:   640×640 pixels
- **Augmentation**: Mosaic, mixup, HSV adjustments, flips

## 📊 Performance

### Best Model Results
- **mAP@50**:    86.4%
- **Precision**: 89.4%
- **Recall**:    80.8%
- **mAP@50-95**: 56.1%

### Interpretation
- Detects 9 out of 10 garbage objects correctly (89.4% precision)
- Finds 8 out of 10 actual garbage items (80.8% recall)
- Excellent for street-level garbage localization

## 🛠️ Installation

```bash
pip install ultralytics torch torchvision

💡 Usage
Training

from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on garbage dataset
results = model.train(
    data='garbage.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='StreetGarbageDetection'
)

Inference
# Detect garbage in images
model = YOLO('best.pt')
results = model.predict(
    source='street_images/',
    conf=0.25,
    save=True
)

Validation
# Evaluate model performance
metrics = model.val(data='garbage.yaml')
print(f"mAP@50: {metrics.box.map50:.3f}")

🎯 Applications
Street cleanliness monitoring

Automated waste detection systems

Municipal garbage collection optimization

Real-time surveillance camera analysis

📈 Future Improvements
Multi-class detection (plastic, paper, metal, etc.)

Larger YOLOv8 variants (s, m, l) for better accuracy

Real-world video stream testing

Integration with garbage collection routes

