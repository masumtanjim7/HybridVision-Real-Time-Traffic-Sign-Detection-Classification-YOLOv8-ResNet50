# HybridVision: Real-Time Traffic Sign Detection & Classification (YOLOv7 + ResNet50)

HybridVision is a **real-time traffic sign recognition system** built with a **hybrid deep learning pipeline** for both **detection** and **fine-grained classification**, specially focused on **Bangladeshi road environments**.  
It combines a YOLO detector to localize signs (bounding boxes) and a ResNet50 classifier to identify the exact traffic sign category from cropped sign regions. A confidence-based fusion strategy is used to produce stable final predictions.

✅ **Custom Dataset:** 9,000+ images (Bangladeshi traffic signs)  
✅ **Total Classes:** 29 sign categories  
✅ **Trained on:** Kaggle GPU (P100)  
✅ **Results (reported):** **mAP 100% / Precision 100% / Recall 100% / Accuracy 99%**  
✅ **Deployed as Web App:** Streamlit (Image / Video / Live Camera)

---

## Table of Contents
- [Why HybridVision?](#why-hybridvision)
- [Key Features](#key-features)
- [Dataset Creation & Annotation](#dataset-creation--annotation)
- [Models Tested (Experiments)](#models-tested-experiments)
- [Final Hybrid Pipeline](#final-hybrid-pipeline)
- [System Workflow](#system-workflow)
- [Fusion Strategy (Decision Logic)](#fusion-strategy-decision-logic)
- [Results](#results)
- [Streamlit Web Interface](#streamlit-web-interface)
- [Repository Structure (Suggested)](#repository-structure-suggested)
- [Installation & Running](#installation--running)
- [Training Notes (Kaggle)](#training-notes-kaggle)
- [Deployment Notes](#deployment-notes)
- [Future Vision (Portfolio-ready)](#future-vision-portfolio-ready)
- [Limitations & Next Improvements](#limitations--next-improvements)
- [Tech Stack](#tech-stack)
- [Screenshots / Demo](#screenshots--demo)
- [Author](#author)
- [License](#license)

---

## Why HybridVision?
Traffic sign recognition has two major problems in real-world roads:
1. **Detection**: signs can be small, blurred, low-light, or partially hidden
2. **Classification**: many signs look similar, especially in crowded backgrounds

YOLO is excellent for **fast detection**, but hybrid vision improves reliability by:
- detecting sign region accurately (**YOLO**)
- classifying the cropped sign with high confidence (**ResNet50**)
- applying a fusion rule to output stable predictions

This hybrid design helps especially in:
- small traffic signs
- noisy backgrounds
- similar sign categories
- video predictions where flicker can happen frame-to-frame

---

## Key Features
- ✅ Real-time traffic sign **detection + classification**
- ✅ Built a **custom Bangladeshi dataset** (9,000+ images)
- ✅ **29 sign categories**
- ✅ Experimented with **YOLOv5, YOLOv7, YOLOv8, YOLOv11**
- ✅ ResNet-based classification, finalized **ResNet50**
- ✅ End-to-end ML workflow:
  - dataset creation → preprocessing/augmentation → training → evaluation → packaging → deployment
- ✅ Streamlit Web UI:
  - **Image Upload**
  - **Video Upload**
  - **Live Camera**
- ✅ Deployment-ready artifacts:
  - YOLO weights (`best.pt`, `last.pt`)
  - ResNet weights (`.pth`)
  - label map (`classes.json` / `labels.pkl`)
  - pipeline bundle (`.pkl`)

---

## Dataset Creation & Annotation
One of the strongest parts of this project is that the dataset is **custom built** for Bangladeshi roads.

### Dataset Workflow
1. **Data Collection**
   - Collected Bangladeshi traffic sign images from multiple sources (real-world road images + curated datasets)
2. **Cleaning & Organization**
   - Removed duplicates and low-quality unusable samples where needed
3. **Roboflow Upload**
   - Used Roboflow to manage dataset versions
4. **Annotation**
   - Labeled traffic signs using bounding boxes
5. **Augmentation**
   - Used augmentation (rotation, blur, brightness/contrast, scaling, etc.) to improve robustness
6. **Export**
   - Exported in YOLO format with Train/Valid/Test split

✅ **Total Images:** 9,000+  
✅ **Total Classes:** 29 categories  
✅ **Goal:** realistic Bangladeshi road scenarios (crowded, low-light, partial occlusion)

---

## Models Tested (Experiments)
To ensure the best final system, I tested multiple model versions instead of using only one model.

### YOLO Detectors Tested
- YOLOv5
- YOLOv7
- YOLOv8
- YOLOv11

### Classifiers Tested
- ResNet-based classifiers (finalized ResNet50)

✅ Final hybrid choice used: **YOLOv7 + ResNet50** *(as the main hybrid model)*  
> If you used YOLOv8 in some experiments, mention it as “also tested” (already included above).

---

## Final Hybrid Pipeline
### Final Setup
- **Detector:** YOLOv7 (real-time bounding boxes)
- **Classifier:** ResNet50 (cropped sign classification)
- **Fusion:** confidence-based final decision

### Why YOLO + ResNet?
- YOLO = fast and accurate localization
- ResNet50 = strong feature extraction and fine classification on cropped regions

---

## System Workflow
### Input Modes
- Image
- Video
- Live Webcam / Camera

### Output
For each detected sign:
- bounding box (x1, y1, x2, y2)
- final predicted label
- confidence score (YOLO + ResNet fusion)
- optionally: both YOLO label and ResNet label for debugging/analysis

### End-to-End Workflow (Project Level)
Dataset creation → preprocessing & augmentation → model training (Kaggle GPU) → evaluation → deployment packaging → Streamlit web app

---

## Fusion Strategy (Decision Logic)
HybridVision uses confidence-based fusion to produce stable predictions.

## Training Notes (Kaggle)

Training was done using **Kaggle GPU (P100)** for speed and reproducibility.

### Training Outputs (Typical)

#### YOLO Training
- `runs/train/.../weights/best.pt`
- `runs/train/.../weights/last.pt`

#### ResNet50 Training
- Saved `.pth` checkpoint (best model)

#### Label Mapping
- `classes.json` or `labels.pkl`

> **Important:** Always save **weights + label map** to make deployment easy and error-free.


## Future Vision (Portfolio-ready)

HybridVision’s long-term vision is to grow into a complete **perception + decision module** for autonomous driving systems in **Bangladeshi road environments**. Beyond real-time traffic sign detection and fine-grained classification, the next goal is to expand the model into a full **road-understanding pipeline** that can support safe driving actions.

### Planned Future Upgrades
- **Traffic light detection** for stop/go decisions at intersections  
- **Lane detection + lane keeping** to maintain stable driving within lanes  
- **Vehicle + pedestrian detection** for collision-aware navigation  
- **Object tracking in video** to reduce flickering detections and keep predictions stable frame-to-frame  
- **A decision module (rule-based first)** to convert detections into driving actions  
  - Example logic: if **“Stop”** sign detected → slow down → stop → wait ~3 seconds → move only if clear  
- **Simulation testing using CARLA** to validate driving behavior safely without real-road risk  
- **Edge deployment optimization** for real-time performance by converting models to **ONNX / TensorRT / TFLite**

## Author

**Md. Masum Billah**  
CSE Graduate | Flutter • IoT • Machine Learning • Computer Vision  

- **Portfolio:** https://masumtanjim7.github.io/portfolio/  
- **GitHub:** https://github.com/masumtanjim7  
