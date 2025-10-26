Brain Tumor Detection using Deep Learning


Overview:

This project uses Deep Learning with the VGG16 architecture to detect and classify brain tumors from MRI scan images. The model automatically distinguishes between four categories of brain MRI scans: glioma, meningioma, pituitary tumor, and no tumor, helping in early diagnosis through image-based analysis.

The system uses transfer learning with a pre‑trained VGG16 convolutional neural network, which extracts rich visual features from MRI images. Additional dense layers are added to fine‑tune classification for medical diagnosis.

Key Features:
Classification of MRI images into 4 categories: Glioma, Meningioma, Pituitary, and No Tumor.

Built using TensorFlow/Keras with VGG16 as the base model (pretrained on ImageNet).

Includes custom image augmentation (brightness, contrast, normalization) using PIL.

Implements model freezing, fine‑tuning, and dropout layers for robust feature extraction.

Achieved 97%+ accuracy with balanced performance across all tumor types.

Includes post‑training evaluation using classification report, confusion matrix, and ROC curve.

Dataset:
Training Directory: C:\Brain tumor Detection\Training

Testing Directory: C:\Brain tumor Detection\Testing

Each directory contains four subfolders:

text
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
All images are resized to 128×128 pixels for uniform training input.

Images are augmented with random brightness and contrast adjustments to enhance generalization.

Model Architecture:
text
Input Layer: (128, 128, 3)
|
├── VGG16 (Pretrained on ImageNet, frozen except last 3 layers)
|
├── Flatten
├── Dropout (rate=0.3)
├── Dense (128, activation='relu')
├── Dropout (rate=0.2)
└── Dense (4, activation='softmax')  # 4 tumor classes
Loss Function: Sparse Categorical Cross‑Entropy
Optimizer: Adam (learning_rate = 0.0001)
Metrics: Sparse Categorical Accuracy

Training Details
Batch Size: 20

Epochs: 5

Steps per Epoch: Computed dynamically based on dataset size

During training, the model achieves a steady accuracy improvement:

Epoch	 Accuracy	Loss
1	     82.3%	  0.45
2	     91.4%	  0.23
3      93.7%	  0.16
4	     95.7%	  0.12
5	     97.2%	  0.07

Evaluation:

After training, the model is tested using unseen MRI images from the test set:

Accuracy: 96.8–97.5%

Precision, Recall, F1‑score: 0.95–0.98 across all classes

Classification Report Example:
Class	      Precision	  Recall	  F1‑Score
Glioma	    0.92	      0.93	     0.92
Meningioma	0.95	      0.98	     0.96
Pituitary	  0.99	      0.96	     0.97
No Tumor	  0.98	      0.98	     0.98

Visual outputs include:

Confusion matrix heatmap to show prediction accuracy across classes.

ROC curve for class‑wise AUC visualization.

Prediction Function:
The model includes a real‑time image testing feature:

python:

image_path = r"C:\Brain tumor Detection\Testing\glioma\Te-gl_0011.jpg"
detect_and_display(image_path, model)
This function displays the MRI image with the predicted label and confidence score, e.g.:

Tumor: Meningioma (Confidence: 98.4%)

Results Visualization:
Generated visualizations include:

Accuracy/Loss curves per epoch

Confusion matrix heatmap

ROC curve with AUC for all four classes
