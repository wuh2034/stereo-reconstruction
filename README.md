# Stereo Reconstruction Project

## Overview
This project implements a stereo reconstruction pipeline in C++ using the OpenCV and Eigen libraries. It features two rectification methods (uncalibrated and calibrated) combined with three keypoint matching algorithms (ORB, SIFT, and BRISK). The pipeline processes stereo image pairs through feature extraction, matching, fundamental matrix estimation, image rectification, and disparity computation to ultimately generate rectified images, disparity maps, depth maps, as well as 3D point clouds and mesh models.

## File Structure
- **stereo-reconstruction.cpp**: The main program file containing the entire stereo reconstruction pipeline.
- **includes/common.hpp**: Contains common functions and data structures (e.g., RANSAC-based fundamental matrix estimation, warp perspective functions, etc.).
- **includes/depth_map_generation.h**: Declarations and implementations for depth map generation functions.
- **includes/mesh_generation.h**: Declarations and implementations for 3D mesh generation functions.

## Dependencies
- **OpenCV**: For image processing, feature detection, matching, rectification, and disparity computation.
- **Eigen**: For matrix operations and linear algebra.
- A compiler supporting **C++11** or later.

## Compilation
It is recommended to use CMake for compilation. Example steps:
```bash
mkdir build
cd build
cmake ..
make
```

# Running the Project

When you run the compiled executable, the program will automatically perform the following steps:

## Dataset Configuration

The program supports two datasets: **Shopvac_imperfect** and **artroom1**. Depending on the chosen dataset, it automatically sets:
- **Image paths** (`leftImagePath` and `rightImagePath`)
- **Camera intrinsic matrices** (`K0` and `K1`)
- **Distortion parameters** (`D0` and `D1`, set to zeros in this case)
- **Focal length and baseline** (used for depth calculation)

## Feature Extraction and Matching

The left and right images are first converted to grayscale. Then, keypoints are detected and descriptors computed using **ORB**, **SIFT**, and **BRISK**. KNN matching is performed followed by a ratio test to filter good matches.

## Fundamental Matrix Estimation

The **RANSAC** algorithm is applied to the matched point pairs to estimate the Fundamental Matrix (F) and count the inliers.

## Rectification and Disparity Map Computation

### Uncalibrated Rectification

- Uses `cv::stereoRectifyUncalibrated` to compute homography matrices for both images.
- The images are warped accordingly, concatenated side-by-side, and horizontal reference lines are drawn for visualization.
- CLAHE is applied for contrast enhancement, followed by disparity map computation using OpenCV's **StereoSGBM** algorithm.
- The resulting disparity map is normalized to an 8-bit image.

### Calibrated Rectification (Custom Implementation)

- An Essential Matrix is computed from F, which is then decomposed to obtain the rotation and translation.
- Projection matrices are built using the camera intrinsics, and custom warp perspective functions (e.g., `myWarpPerspective`) are used for rectification.
- A custom **FastSGBM** algorithm computes the disparity map.

### Calibrated Rectification (OpenCV Built-in)

- Uses OpenCV's `cv::stereoRectify` function to rectify the images.
- Followed by remapping (`cv::initUndistortRectifyMap` and `cv::remap`) and disparity computation via **StereoSGBM**.

## Depth Map and 3D Mesh Generation

For the **BRISK** method, the computed disparity maps are used to generate depth maps. These depth maps are further processed to produce point clouds and 3D mesh models (saved in `.off` format) in a designated results folder (e.g., `/workspace/results/`).

## Output Files

### Rectified Images

The rectified images are saved with filenames such as:
- **Uncalibrated**: `uncalib_rectify_opencv_SIFT.png`
- **Custom calibrated**: `ours_calib_rectify_BRISK.png`
- **OpenCV calibrated**: `calib_opencv_ORB.png`

### Disparity Maps

Disparity maps are saved with filenames such as:
- **Uncalibrated**: `uncalib_disparity_opencv_SIFT.png`
- **Custom calibrated**: `ours_calib_disparity_BRISK.png`
- **OpenCV calibrated**: `calib_disparity_opencv_ORB.png`

### Depth Maps and 3D Mesh Models (for BRISK method only)

- **Depth maps**: e.g., `/workspace/results/Shopvac_imperfect_depth_map_uncalib_opencv_BRISK.png`
- **Mesh files**: e.g., `/workspace/results/Shopvac_imperfect_mesh_uncalib_opencv_BRISK.off`

## Parameter Adjustments

### Switching Datasets

Modify the dataset identifier in `stereo-reconstruction.cpp` (e.g., `"Shopvac_imperfect"` or `"artroom1"`), and update the corresponding image paths and camera intrinsic matrices as needed.

### Depth Map and Mesh Generation Parameters

Adjust the distance thresholds (`upperDistanceThreshold` and `lowerDistanceThreshold`) to suit the specific dataset and desired mesh quality.

## Notes

- Ensure that the image paths and dataset configurations are correct; otherwise, the program will report errors when loading images.
- Some functions (e.g., `ransacFundamentalMatrix`, `decomposeEssentialMatrix`, `myWarpPerspective`) are implemented in separate header/source files—make sure these files are available in your build.
- Depending on dataset and application scenario, might need to fine-tune algorithm parameters (such as the number of RANSAC iterations or disparity computation parameters).

