# PLANT-DISEASE-DITECTION

# 🥔 Potato Leaf Disease Detection Using YOLOv8 and EfficientNetV2

## 📌 Project Overview

Potato plants are affected by diseases such as Early Blight and Late Blight, which can reduce crop quality and agricultural productivity. Identifying these diseases at an early stage can help farmers take appropriate action.

This project develops a deep learning-based potato leaf disease detection system using a hybrid architecture that combines **YOLOv8 for disease-region detection** and **EfficientNetV2S for disease classification**.

The system can process images and video frames, detect relevant regions, classify the detected regions, and display the predicted disease label with a confidence score.

## 🎯 Project Objectives

- Detect disease-affected regions in potato leaf images.
- Classify detected regions into disease categories.
- Support image-based disease detection.
- Process video files frame by frame.
- Support webcam-based detection.
- Compare object detection and image classification approaches.
- Build a reusable deep learning workflow for potato leaf disease analysis.

## 🦠 Disease Categories

The code uses four class labels:

1. Early Blight
2. Late Blight
3. Healthy
4. Other

**Note:** The class order must match the class indices used when training both models. The `Other` class is included in the supplied code, but its exact dataset definition should be documented separately.

## 🧠 Model Architecture

### 1. YOLOv8 – Disease Region Detection

YOLOv8 is used to detect relevant regions in potato leaf images.

The supplied YOLO training scripts use:

- Base model: `yolov8m.pt`
- Optimizer: AdamW
- Learning rate: 0.0007
- Image size: 832 in one training configuration
- Epochs: 150 in that configuration
- Dataset configuration: `data.yaml`
- Validation: `model.val()`

A second YOLO training configuration uses `yolov8m.pt`, 416 image size, and 30 epochs. These are separate experimental configurations, not one combined training run.

YOLOv8 produces bounding boxes and detection confidence scores for the detected regions.

### 2. EfficientNetV2S – Disease Classification

EfficientNetV2S is used to classify cropped image regions into the four disease categories.

The supplied training script uses:

- Architecture: EfficientNetV2S
- Pretrained weights: ImageNet
- Input image size: 224 × 224
- Batch size: 16
- Number of classes: 4
- Optimizer: AdamW
- Loss function: Categorical Crossentropy
- Evaluation metric: Accuracy

The classifier uses a custom classification head containing:

- Global Average Pooling
- Batch Normalization
- Dense layer with 512 units and ReLU activation
- Dropout with a rate of 0.4
- Softmax output layer

### 3. Hybrid YOLOv8 + EfficientNetV2 Pipeline

The hybrid detection process follows these steps:

1. Load the input image or video frame.
2. Run YOLOv8 inference to detect relevant regions.
3. Extract the bounding-box coordinates.
4. Crop the detected region from the original image.
5. Resize the crop to 224 × 224.
6. Convert the crop from BGR to RGB.
7. Apply EfficientNetV2 preprocessing.
8. Run the crop through the trained classifier.
9. Select the class with the highest predicted probability.
10. Display the disease label and confidence score.

The supplied hybrid scripts use a detection confidence threshold of 0.25 and a classification confidence threshold of 0.60 by default.

## ⚙️ Training Configuration

### YOLOv8 Training

The `training yolo 02.py` script contains the following configuration:

| Parameter | Value |
|---|---|
| Base model | YOLOv8m |
| Epochs | 150 |
| Image size | 832 |
| Optimizer | AdamW |
| Initial learning rate | 0.0007 |
| Weight decay | 0.0003 |
| Warmup epochs | 5 |
| Batch size | Automatic (`-1`) |
| Device | GPU 0 |
| Validation | Enabled |

Disease-aware augmentation settings include:

- Horizontal flipping
- Small translations
- Scaling
- HSV adjustments
- Low mosaic augmentation
- No MixUp
- No Copy-Paste

### EfficientNetV2S Training

The `training cnn 01.py` script uses a two-stage training approach.

**Stage 1 – Transfer Learning**
- Load ImageNet-pretrained EfficientNetV2S.
- Freeze the base model.
- Train the custom classification head.
- Train for 20 epochs.

**Stage 2 – Fine-Tuning**
- Unfreeze the base model.
- Keep the earlier layers frozen.
- Allow the final 40 layers to become trainable.
- Recompile using a lower learning rate.
- Continue training with a maximum configured epoch count of 60.

Callbacks used:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

## 🖼️ Supported Detection Modes

### Image Detection

The system accepts a potato leaf image, detects disease regions, classifies the detected crops, and displays an annotated image.

### Video Detection

The system reads a video frame by frame, performs detection and classification, and displays the processed video with disease labels.

### Webcam Detection

The video-processing function accepts `0` as the video source to use the default webcam.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | EfficientNetV2S training and classification |
| Ultralytics YOLOv8 | Disease-region detection |
| OpenCV | Image and video processing |
| NumPy | Array operations |
| ImageDataGenerator | Image loading and augmentation |
| Matplotlib | Optional training visualization |

## 📂 Project Structure

The following is a recommended organized structure based on the supplied scripts. Adjust the folder names to match your actual project directory.

```text
Potato-Leaf-Disease-Detection/
│
├── Datasets/
│   └── Potato_Disease/
│       ├── data.yaml
│       ├── train/
│       ├── val/
│       └── test/
│
├── Potato_Disease_Classification/
│   ├── train/
│   │   ├── Early Blight/
│   │   ├── Late Blight/
│   │   ├── Healthy/
│   │   └── Other/
│   └── val/
│       ├── Early Blight/
│       ├── Late Blight/
│       ├── Healthy/
│       └── Other/
│
├── Model/
│   ├── PotatoDisease/
│   │   └── best.pt
│   └── EfficientNetV2_PotatoDisease.h5
│
├── Samples/
│   ├── Potato Leaf02.jpg
│   └── Potato Disease02.mp4
│
├── training yolo 02.py
├── training cnn 01.py
├── Detection img 01.py
├── Detection vdo01.py
├── Detect01.py
├── Detect02.py
├── Tranning01.py
├── requirements.txt
└── README.md
```

## 💻 Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip package manager.
- GPU with compatible CUDA dependencies for GPU training (optional).
- Sufficient RAM and storage for the datasets and trained models.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Potato-Leaf-Disease-Detection
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Install Dependencies

```bash
pip install ultralytics tensorflow opencv-python numpy matplotlib
```

Alternatively:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Datasets

- Organize the YOLO dataset into training, validation, and testing folders.
- Configure the YOLO dataset paths and class names in `data.yaml`.
- Organize the classification dataset into class-specific folders under `train` and `val`.
- Ensure that the four classification folder names match the expected class labels.

## ▶️ Running the Project

### Train YOLOv8

```bash
python "training yolo 02.py"
```

### Train EfficientNetV2S

```bash
python "training cnn 01.py"
```

### Run Image Detection

```bash
python "Detection img 01.py"
```

### Run Video Detection

```bash
python "Detection vdo01.py"
```

For webcam detection, use:

```python
run_video(0)
```

The hybrid image and video scripts require both trained model files to be available at the paths configured in the scripts.

## 📊 Model Evaluation

The YOLO training script calls:

```python
metrics = model.val()
print(metrics)
```

The EfficientNetV2S training script monitors validation accuracy and saves the best model checkpoint.

Evaluation can include:

- Classification accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- YOLO precision and recall
- Mean Average Precision (mAP)

Actual evaluation values should be added after running the final training and validation process. No performance scores are claimed in this README.

## 🔮 Future Improvements

- Add a web application using Flask or Streamlit.
- Create a farmer-friendly interface for uploading leaf images.
- Add disease severity estimation.
- Add treatment and prevention recommendations from verified agricultural sources.
- Improve performance on different lighting conditions and leaf backgrounds.
- Evaluate the model on independent field images.
- Add model explainability and prediction reports.
- Optimize inference speed for deployment on edge devices.

## ⚠️ Limitations

- Predictions depend on dataset quality and diversity.
- Performance on field images may differ from performance on training or validation data.
- The system is designed for the four configured classes.
- Detection and classification confidence thresholds affect which results are displayed.
- The model should not be treated as a substitute for expert agricultural diagnosis.

## 👤 Author

**Nilay Ghosh**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/nilay3ghosh

## 📝 Project Purpose

This project was developed for academic and learning purposes to explore deep learning, object detection, transfer learning, image classification, and computer vision applications in agriculture.


# Potato Leaf Disease Detection – 60 Interview Questions and Answers

## Section 1: Project Introduction

**Q1. Explain your Potato Leaf Disease Detection project.**

**Answer:**  
My project is a deep learning-based potato leaf disease detection system developed using YOLOv8 and EfficientNetV2S. YOLOv8 detects disease-related regions in leaf images, while EfficientNetV2S classifies the detected regions into Early Blight, Late Blight, Healthy, and Other. I also implemented image and video processing using OpenCV.

**Q2. What is the main objective of this project?**

**Answer:**  
The main objective is to automate the identification of potato leaf diseases using computer vision and deep learning. The system aims to detect affected regions and classify them into predefined disease categories.

**Q3. Which diseases can your system identify?**

**Answer:**  
The code uses four classes: Early Blight, Late Blight, Healthy, and Other.

**Q4. Why did you choose potato leaf disease detection?**

**Answer:**  
Potato is an important agricultural crop, and leaf diseases can affect plant health and productivity. Automated image-based detection can help identify possible disease symptoms more efficiently and support early investigation.

**Q5. Is your project based on machine learning or deep learning?**

**Answer:**  
It is a deep learning project. It uses convolutional neural network-based architectures, including YOLOv8 and EfficientNetV2S.

**Q6. Which technologies did you use?**

**Answer:**  
Python, TensorFlow, Keras, Ultralytics YOLOv8, OpenCV, and NumPy.

**Q7. What are the main modules of your project?**

**Answer:**
1. YOLOv8 model training.
2. EfficientNetV2S model training.
3. Image-based detection.
4. Video-based detection.
5. Hybrid detection and classification.

**Q8. What is the difference between object detection and image classification?**

**Answer:**  
Image classification assigns a label to an entire image. Object detection identifies objects or regions within an image and provides their bounding-box coordinates and class predictions.

**Q9. Why did you use two models?**

**Answer:**  
The hybrid approach separates localization and classification. YOLOv8 identifies relevant regions, and EfficientNetV2S performs classification on the cropped regions. This allows the system to use an object detector and a dedicated image classifier for different tasks.

**Q10. What are the outputs of your system?**

**Answer:**  
The system displays an annotated image or video frame containing a bounding box, predicted disease label, and classification confidence score.

## Section 2: YOLOv8 and Object Detection

**Q11. What is YOLO?**

**Answer:**  
YOLO stands for You Only Look Once. It is an object detection family that predicts object locations and classes in an image using a single network inference process.

**Q12. What is YOLOv8?**

**Answer:**  
YOLOv8 is an object detection model family from Ultralytics. It supports object detection and other computer vision tasks.

**Q13. Why did you use YOLOv8?**

**Answer:**  
I used YOLOv8 because it can detect regions of interest and provide bounding-box coordinates with confidence scores. These detected regions can then be cropped for further classification.

**Q14. What is a bounding box?**

**Answer:**  
A bounding box is a rectangular region around a detected object. In my project, it identifies the detected leaf region that is passed to the classification model.

**Q15. What does `best.pt` represent?**

**Answer:**  
`best.pt` is a saved YOLO model checkpoint containing the selected trained model weights. In the Ultralytics training workflow, it generally represents the best checkpoint according to the configured validation fitness criterion.

**Q16. What is the purpose of `data.yaml`?**

**Answer:**  
The YOLO dataset configuration file defines dataset paths and class information, such as the training and validation image locations and class names.

**Q17. What is the confidence threshold in YOLO?**

**Answer:**  
The confidence threshold determines the minimum detection confidence required for a predicted bounding box to be retained. My detection scripts use a default threshold of 0.25.

**Q18. What is IoU?**

**Answer:**  
IoU stands for Intersection over Union. It measures the overlap between a predicted bounding box and a ground-truth bounding box.

Formula:

IoU = Area of Intersection / Area of Union

**Q19. What is Non-Maximum Suppression?**

**Answer:**  
Non-Maximum Suppression, or NMS, removes duplicate or highly overlapping detections by retaining stronger predictions according to confidence and overlap criteria.

**Q20. What is the difference between YOLOv8n, YOLOv8s, YOLOv8m, and YOLOv8l?**

**Answer:**  
They are different model sizes. Nano and small versions are generally lighter and faster, while medium and large versions generally require more computation and can offer greater model capacity. The best choice depends on hardware, dataset, and validation performance.

**Q21. Which YOLO model did you train?**

**Answer:**  
The supplied training scripts use `yolov8m.pt` as the starting model.

**Q22. What does `imgsz=832` mean?**

**Answer:**  
It sets the training image size to 832 pixels for the YOLO training configuration. A larger image size can help preserve small visual details, but it also increases computational and memory requirements.

**Q23. What does `epochs=150` mean?**

**Answer:**  
It means the model is configured to train for up to 150 epochs. One epoch represents one complete pass through the training dataset.

## Section 3: EfficientNetV2S and CNN

**Q24. What is a CNN?**

**Answer:**  
CNN stands for Convolutional Neural Network. It is a deep learning architecture commonly used for image processing. It learns visual patterns such as edges, textures, shapes, and more complex features through convolutional layers.

**Q25. What is EfficientNetV2S?**

**Answer:**  
EfficientNetV2S is a convolutional neural network architecture designed to provide an efficient balance between model performance and computational cost. My project uses its ImageNet-pretrained weights for transfer learning.

**Q26. Why did you use EfficientNetV2S?**

**Answer:**  
I used EfficientNetV2S as the classification component because it provides a pretrained feature extractor that can be adapted to classify potato leaf disease categories.

**Q27. What is transfer learning?**

**Answer:**  
Transfer learning is a technique in which a model trained on a large dataset is reused for a different but related task. In my project, EfficientNetV2S starts with ImageNet-pretrained weights and is adapted to potato disease classification.

**Q28. What is fine-tuning?**

**Answer:**  
Fine-tuning involves unfreezing some pretrained model layers and training them further on the target dataset so that the model can adapt its learned features to the new task.

**Q29. What is the difference between feature extraction and fine-tuning?**

**Answer:**  
Feature extraction keeps the pretrained backbone frozen and trains a new classification head. Fine-tuning allows selected pretrained layers to update their weights using the new dataset.

**Q30. Why did you freeze the base model initially?**

**Answer:**  
Freezing the base model preserves its pretrained features while the new classification head learns the target classes. It also reduces the number of trainable parameters during the initial training stage.

**Q31. What is Global Average Pooling?**

**Answer:**  
Global Average Pooling calculates the average value of each feature map, converting spatial feature maps into a compact feature vector. It reduces the number of parameters compared with flattening the entire feature map.

**Q32. What is Dropout?**

**Answer:**  
Dropout is a regularization technique that randomly disables a fraction of neurons during training. It helps reduce overfitting.

**Q33. What is Batch Normalization?**

**Answer:**  
Batch Normalization normalizes intermediate activations during training. It can improve training stability and help neural networks converge more effectively.

**Q34. Why did you use Softmax in the final layer?**

**Answer:**  
Softmax converts the model's output scores into a probability distribution across the four classes. The class with the highest predicted probability is selected as the prediction.

**Q35. Why did you use categorical cross-entropy?**

**Answer:**  
Categorical cross-entropy is a loss function commonly used for multi-class classification when the target labels are represented in categorical or one-hot encoded form.

## Section 4: Image Processing and Hybrid Architecture

**Q36. What is OpenCV?**

**Answer:**  
OpenCV is an open-source computer vision library. I used it to load images, process video frames, crop detected regions, draw bounding boxes, display labels, and show the output.

**Q37. What is the role of NumPy?**

**Answer:**  
NumPy provides numerical array operations. In my project, it is used for image array manipulation, expanding dimensions, and selecting the predicted class using `argmax()`.

**Q38. Why do you resize the cropped region to 224 × 224?**

**Answer:**  
The EfficientNetV2S classifier is configured with an input shape of 224 × 224 × 3. Resizing ensures that each cropped region has the required dimensions before classification.

**Q39. Why do you convert BGR to RGB?**

**Answer:**  
OpenCV loads images in BGR channel order by default, whereas the classification pipeline expects RGB image channels. Therefore, I convert the image using `cv2.COLOR_BGR2RGB`.

**Q40. What is `preprocess_input()`?**

**Answer:**  
It is the preprocessing function associated with EfficientNetV2. It prepares the input image in the format expected by the model.

**Q41. Explain the hybrid architecture step by step.**

**Answer:**
1. Load an image or video frame.
2. Run YOLOv8 detection.
3. Extract bounding-box coordinates.
4. Crop each detected region.
5. Resize the crop to 224 × 224.
6. Convert the crop to RGB.
7. Apply EfficientNetV2 preprocessing.
8. Classify the crop.
9. Select the highest-probability class.
10. Draw the final disease label and confidence on the image.

**Q42. What is the purpose of `np.argmax()`?**

**Answer:**  
`np.argmax()` returns the index of the highest value in an array. In my classification code, it identifies the class with the highest predicted probability.

**Q43. What is the difference between detection confidence and classification confidence?**

**Answer:**  
Detection confidence is produced by YOLOv8 and indicates confidence in a detected region. Classification confidence is produced by EfficientNetV2S and indicates the model's predicted probability for the selected disease category.

**Q44. Why do you use two confidence thresholds?**

**Answer:**  
The detection threshold filters YOLOv8 detections, while the classification threshold filters disease predictions from EfficientNetV2S. My hybrid code uses default values of 0.25 and 0.60, respectively.

## Section 5: Training and Optimization

**Q45. What is data augmentation?**

**Answer:**  
Data augmentation creates modified training images using transformations such as rotation, shifting, zooming, brightness adjustment, and flipping. It helps increase training data variation and can improve generalization.

**Q46. Which augmentations did you use for EfficientNetV2S?**

**Answer:**  
The supplied CNN training script uses rotation, width and height shifts, zoom, brightness adjustments, and horizontal flipping.

**Q47. Which augmentations did you use for YOLOv8?**

**Answer:**  
The supplied YOLO training configuration uses small translations, scaling, HSV adjustments, horizontal flipping, light erasing, and low mosaic augmentation. MixUp and Copy-Paste are disabled.

**Q48. What is AdamW?**

**Answer:**  
AdamW is an optimization algorithm based on Adam that separates weight decay from the gradient-based parameter updates. It is used to optimize the model's trainable parameters.

**Q49. What is learning rate?**

**Answer:**  
Learning rate controls the size of the parameter updates during model training. A very high learning rate may cause unstable training, while a very low learning rate may make training slow.

**Q50. What is EarlyStopping?**

**Answer:**  
EarlyStopping monitors a selected validation metric and stops training when the metric stops improving for a specified number of epochs. In my CNN training script, it uses a patience value of 10 and restores the best weights.

**Q51. What is ReduceLROnPlateau?**

**Answer:**  
ReduceLROnPlateau reduces the learning rate when a monitored metric stops improving. This can help the model make smaller updates during later training.

**Q52. What is ModelCheckpoint?**

**Answer:**  
ModelCheckpoint saves model weights or the model itself when the monitored metric reaches a new best value. My CNN script saves the best model based on validation accuracy.

## Section 6: Evaluation and Practical Scenarios

**Q53. Which evaluation metrics are useful for your project?**

**Answer:**  
For the classification model, useful metrics include accuracy, precision, recall, F1-score, and a confusion matrix. For YOLO object detection, useful metrics include precision, recall, and mean Average Precision (mAP).

**Q54. What is mAP?**

**Answer:**  
Mean Average Precision is an object detection evaluation metric. It summarizes average precision across object classes and, depending on the reported metric, specified IoU thresholds.

**Q55. What is overfitting?**

**Answer:**  
Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data. Data augmentation, dropout, regularization, early stopping, and appropriate validation can help reduce it.

**Q56. What happens if YOLO detects a region incorrectly?**

**Answer:**  
The classifier may receive an irrelevant or incorrectly cropped region. This can lead to an incorrect disease prediction. Improving the detector's training data, annotations, and validation performance can reduce this problem.

**Q57. What happens if the classification confidence is below 0.60?**

**Answer:**  
In the supplied hybrid image and video scripts, predictions below the classification confidence threshold are skipped rather than displayed.

**Q58. How would you improve this project?**

**Answer:**  
I would improve it by collecting more diverse field images, evaluating the models on independent test data, tuning model parameters, checking class imbalance, improving class-wise evaluation, and building a user-friendly web application. I would also investigate disease severity estimation and explainability.

**Q59. What are the limitations of your project?**

**Answer:**  
The model is limited by the quality and diversity of its training data, the four configured classes, detection accuracy, and classification confidence thresholds. Performance on real agricultural field images may differ from performance on the training and validation datasets.

**Q60. How would you explain the project to a non-technical person?**

**Answer:**  
This project uses artificial intelligence to examine potato leaf images and identify possible disease symptoms. It first locates relevant regions in the leaf and then classifies them into categories such as Early Blight, Late Blight, Healthy, and Other. It can also process video frames to display predictions.
