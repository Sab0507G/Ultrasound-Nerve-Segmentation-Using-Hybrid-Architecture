# Ultrasound Nerve Segmentation using U-Net

## Introduction
Ultrasound imaging is a widely used technique in medical diagnostics. However, segmenting anatomical structures, such as nerves, from ultrasound images is a challenging task due to low contrast, speckle noise, and variability in patient anatomy. This project aims to develop a **U-Net-based deep learning model** for **Brachial Plexus segmentation** from ultrasound images, improving accuracy and reliability in nerve detection.

## Dataset
The dataset used for this project is sourced from **Kaggle**, specifically for **Brachial Plexus nerve segmentation**. It consists of ultrasound images with corresponding **pixel-wise segmentation masks**. The masks indicate the regions where the nerve structures are present.

### Dataset Structure:
- **Images**: Ultrasound scans of the Brachial Plexus nerve region.
- **Masks**: Binary segmentation masks marking the Brachial Plexus nerve.
- **CSV File**: Metadata containing pixel-level information for training.

## Model Architecture
The project employs **U-Net**, a popular deep learning model for medical image segmentation. The U-Net consists of:
- **Contracting Path (Encoder)**: Extracts features using convolutional layers.
- **Bottleneck**: Captures high-level representations.
- **Expanding Path (Decoder)**: Restores spatial resolution for precise segmentation.
- **Skip Connections**: Preserve fine details by connecting encoder and decoder layers.

## Methodology
1. **Data Preprocessing:**
   - Convert images to grayscale (if needed).
   - Normalize pixel values.
   - Apply augmentation (rotation, flipping, contrast enhancement).
2. **Model Training:**
   - Train U-Net using **Binary Cross-Entropy Dice Loss**.
   - Optimize with **Adam optimizer**.
   - Implement early stopping to prevent overfitting.
3. **Evaluation Metrics:**
   - **Dice Coefficient**
   - **Jaccard Index (IoU)**
   - **Pixel Accuracy**

## Results
- Achieved **Dice Score of 0.3945%** on validation data.
- Successfully segmented Brachial Plexus nerve structures with high precision.

## Future Work
- Experiment with **Attention U-Net** for better segmentation.
- Apply **post-processing techniques** like CRFs for refinement.
- Extend model for real-time ultrasound video segmentation.

## Conclusion
This project demonstrates the effectiveness of **U-Net for Brachial Plexus nerve segmentation**, contributing towards improved medical imaging analysis. By leveraging deep learning, we can enhance diagnostic accuracy and aid medical professionals in nerve identification.

## Acknowledgments
- Kaggle for providing the dataset.
- Open-source medical imaging communities for research and support.

