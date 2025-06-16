# Computer Vision Assignments

This repository contains five assignments completed for a Computer Vision course, covering various fundamental and advanced computer vision techniques and applications.

## Assignment 1: Image Processing Fundamentals

### Image Cartoonifying
- Implementation of image processing filters to transform regular photographs into cartoon-style images
- Techniques: Edge detection, color quantization, bilateral filtering
- Results available in `asg1/Cartoonifying.ipynb`

### Road Lane Detection
- Detection of road lanes in driving scenarios using Hough Transform
- Pipeline: Grayscale conversion, edge detection, region of interest selection, line detection
- Implementation in `asg1/lane_detection.ipynb`

## Assignment 2: Homography and Image Mosaics

### Augmented Reality with Planar Homographies
- Implementation of augmented reality by projecting images onto planar surfaces
- Feature detection, matching, and homography estimation
- Video overlay with perspective correction
- Results in `asg2/augmented_reality.ipynb`

### Image Mosaics
- Creation of panoramic images by stitching multiple images together
- Feature extraction, matching, and blending techniques
- Implementation in `asg2/image-mosaics.ipynb`

## Assignment 3: Stereo Vision and Depth Estimation

- Implementation of stereo matching algorithms for depth estimation
- Techniques:
  - Block matching with Sum of Absolute Differences (SAD) and Sum of Squared Differences (SSD)
  - Dynamic programming for optimal disparity estimation
- Evaluation of different window sizes and matching costs
- Implementation in `asg3/stereo.ipynb`

## Assignment 4: Object Detection Models

- Comparison of three different object detection architectures:
  - YOLOv5 (one-stage detector)
  - Faster R-CNN (two-stage detector)
  - DETR (Transformer-based detector)
- Evaluation on COCO and Pascal VOC datasets
- Performance metrics: IoU, mAP, inference time
- Feature map visualization and GradCAM for model interpretability
- Implementation in:
  - `asg4/YOLO/YOLO COCO Object Detection.ipynb`
  - `asg4/Faster-RCNN/object-detection-assignment-fast-rcnn.ipynb`
  - `asg4/DETR/DETR-COCO.ipynb` and `asg4/DETR/DETR-PASCAL.ipynb`
  - `asg4/Model_Comparison.ipynb`

## Assignment 5: Object Tracking

- Implementation of Lucas-Kanade optical flow for object tracking
- Template-based tracking with feature point detection and matching
- Tracking of objects in video sequences
- Implementation in `asg5/lk_tracker.ipynb`
- Results available in the `asg5/output` directory

## Repository Structure

Each assignment is contained in its own directory (`asg1`, `asg2`, etc.) with the following structure:
- Jupyter notebooks containing the implementation
- Data directories with input images/videos
- Output directories with results
- PDF reports detailing the methodology and findings

## Requirements

The assignments use various Python libraries including:
- OpenCV
- NumPy
- Matplotlib
- PyTorch (for deep learning-based assignments)
- scikit-image
- Jupyter Notebook

## Usage

1. Navigate to the specific assignment directory
2. Open the Jupyter notebook files to view the implementation and results
3. Follow the instructions in each notebook to run the code

## Reports

Each assignment includes a detailed report (PDF) explaining the methodology, implementation details, and results analysis.