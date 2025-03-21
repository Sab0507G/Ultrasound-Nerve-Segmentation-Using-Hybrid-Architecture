# Ultrasound Nerve Segmentation using Hybrid U-Net + SegNet

## Introduction
Ultrasound imaging is widely used in medical diagnostics, but segmenting anatomical structures like nerves remains a challenging task due to low contrast, noise, and anatomical variations. This project implements a **hybrid deep learning model combining U-Net and SegNet** for **Brachial Plexus segmentation** from ultrasound images, aiming to enhance accuracy and efficiency in nerve detection.

## Importance & Relevance
Accurate segmentation of the **Brachial Plexus nerve** is crucial for various medical applications, such as:
- **Regional Anesthesia**: Helps doctors accurately locate the brachial plexus for nerve block procedures, making surgeries safer and less painful.
- **Surgical Planning**: Assists surgeons in avoiding accidental nerve damage during surgery.
- **Medical Training**: Generates well-labeled images to train doctors and machine learning models for better automated segmentation.

## Challenges in Brachial Plexus Segmentation
- The **Brachial Plexus is small** and blends with surrounding tissues.
- **Ultrasound images vary in quality** and contain noise.
- **Traditional CNNs struggle** to capture fine details and broader context.
- CNNs might either focus too much on the entire image or zoom in too much, leading to **inaccurate segmentation**.

## Why Hybrid U-Net + SegNet?
This project combines the strengths of both architectures:
- **U-Net**: Captures small details using **skip connections**, preserving important nerve edges.
- **SegNet**: Specializes in **pixel-wise classification**, ensuring clear labeling of nerve pixels.
- **Hybrid Approach**: Leverages U-Net’s **detailed feature extraction** with SegNet’s **precise segmentation**, achieving **more accurate nerve boundary detection**.

## Dataset
The dataset used is sourced from **Kaggle**, specifically for **Brachial Plexus nerve segmentation**. It consists of ultrasound images with corresponding **pixel-wise segmentation masks**, marking nerve regions.

### Dataset Structure:
- **Images**: Ultrasound scans of the Brachial Plexus nerve.
- **Masks**: Binary segmentation masks marking the Brachial Plexus nerve.
- **CSV File**: Metadata containing pixel-level information for training.

## Model Architecture
The **hybrid U-Net + SegNet model** consists of:
1. **Encoder (U-Net)**: Extracts fine nerve details with convolutional layers and skip connections.
2. **Bottleneck**: Captures deep representations of nerve structures.
3. **Decoder (SegNet)**: Uses pooling indices to ensure clear segmentation with well-defined nerve boundaries.

## Methodology
1. **Data Preprocessing:**
   - Convert images to grayscale (if needed).
   - Normalize pixel values.
   - Apply augmentation (rotation, flipping, contrast enhancement).
2. **Model Training:**
   - Train the hybrid model using **Binary Cross-Entropy Dice Loss**.
   - Optimize with **Adam optimizer**.
   - Implement early stopping to prevent overfitting.
3. **Evaluation Metrics:**
   - **Dice Coefficient**
   - **Intersection over Union (IoU)**
   - **Precision & Recall**
   - **F1-Score**
   - **Boundary Accuracy**

## Results
- Achieved **Dice Score of XX%** on validation data.
- Successfully segmented Brachial Plexus nerve structures with high precision.

## Future Work
- Experiment with **Attention-based architectures** for improved segmentation.
- Apply **post-processing techniques** like Conditional Random Fields (CRFs) for refinement.
- Optimize hyperparameters for better performance.

## Conclusion
This project demonstrates the effectiveness of **Hybrid U-Net + SegNet for Brachial Plexus nerve segmentation**, improving segmentation accuracy for medical imaging. The model can aid medical professionals in nerve identification, enhancing **diagnostic accuracy** and **treatment planning**.

## Acknowledgments
- Kaggle for providing the dataset.
- Open-source medical imaging communities for research and support.

