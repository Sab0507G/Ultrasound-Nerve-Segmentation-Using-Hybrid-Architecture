# Ultrasound Nerve Segmentation using SegNet

## Introduction
Ultrasound imaging is widely used in medical diagnostics, but segmenting anatomical structures, such as nerves, from ultrasound images remains a challenging task due to low contrast, speckle noise, and anatomical variations. This project implements **SegNet**, a deep learning model for **Brachial Plexus segmentation** from ultrasound images, aiming to enhance accuracy and efficiency in nerve detection.

## Dataset
The dataset used for this project is sourced from **Kaggle**, specifically for **Brachial Plexus nerve segmentation**. It consists of ultrasound images with corresponding **pixel-wise segmentation masks**, which indicate the regions where the nerve structures are present.

### Dataset Structure:
- **Images**: Ultrasound scans of the Brachial Plexus nerve region.
- **Masks**: Binary segmentation masks marking the Brachial Plexus nerve.
- **CSV File**: Metadata containing pixel-level information for training.

## Model Architecture
The project employs **SegNet**, an encoder-decoder deep learning model designed for semantic segmentation. SegNet consists of:
- **Encoder**: A series of convolutional layers that extract high-level features.
- **Pooling Indices**: Used to retain spatial information during downsampling.
- **Decoder**: Restores spatial resolution by upsampling using the stored pooling indices.

## Methodology
1. **Data Preprocessing:**
   - Convert images to grayscale (if needed).
   - Normalize pixel values.
   - Apply augmentation (rotation, flipping, contrast enhancement).
2. **Model Training:**
   - Train SegNet using **Binary Cross-Entropy Dice Loss**.
   - Optimize with **Adam optimizer**.
   - Implement early stopping to prevent overfitting.
3. **Evaluation Metrics:**
   - **Dice Coefficient**
   - **Jaccard Index (IoU)**
   - **Pixel Accuracy**

## Results
- Achieved **Dice Score of 0.397%** on validation data.
- Successfully segmented Brachial Plexus nerve structures with reasonable precision.

## Future Work
- Experiment with **Attention SegNet** for improved segmentation.
- Apply **post-processing techniques** such as CRFs for refinement.
- Optimize hyperparameters for better performance.

## Conclusion
This project demonstrates the effectiveness of **SegNet for Brachial Plexus nerve segmentation**, contributing towards improved medical imaging analysis. By leveraging deep learning, we can enhance diagnostic accuracy and aid medical professionals in nerve identification.

## Acknowledgments
- Kaggle for providing the dataset.
- Open-source medical imaging communities for research and support.

