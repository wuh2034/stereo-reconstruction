//
// Created by Yidi Ma on 02.02.25.
//

#ifndef DEPTH_MAP_GENERATION_H
#define DEPTH_MAP_GENERATION_H

#endif //DEPTH_MAP_GENERATION_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <cmath>
#include <opencv2/ximgproc.hpp>
#include <fstream>
#include <arpa/inet.h>

inline cv::Mat computeDepthMapAndVis(
    const cv::Mat& disparity, double focalLength, double baseline,
    const std::string& outputPath,
    const float upperDistanceThreshold, const float lowerDistThreshold)
{

    float threshold = 0.01;

    // using the absolute disparity map to only consider the magnitude of disparity and not
    // the direction -> can lead to false negatives for (negative) depth values
    cv::Mat disparity_abs = abs(disparity);

    // mask out all valid disparity values
    cv::Mat mask = disparity_abs > threshold;

    // copy depth map to new valid depth map
    cv::Mat depth = cv::Mat::zeros(disparity_abs.size(), CV_32F);
    cv::Mat validDisparity;
    disparity_abs.copyTo(validDisparity, mask);

    // compute valid depth values from valid disparity values
    cv::divide(baseline * focalLength, validDisparity, depth, 1.0, CV_32F);

    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    std::cout << "[DEPTH] Min value depth in mm before filtering: " << minVal << std::endl;
    std::cout << "[DEPTH] Max value depth in mm before filtering: " << maxVal << std::endl;

    // Set invalid/infinite depth values to 0
    // count number of 0 values in depth map
    depth = depth / 1000.0f;
    depth.setTo(0, ~mask);  // Set masked-out areas to 0
    depth.setTo(0, depth == std::numeric_limits<float>::infinity());
    depth.setTo(0, depth < lowerDistThreshold);    // closer than 20cm
    depth.setTo(0, depth > upperDistanceThreshold);  // farther than 20m

    cv::minMaxLoc(depth, &minVal, &maxVal);
    std::cout << "[DEPTH] Min value depth in m: " << minVal << std::endl;
    std::cout << "[DEPTH] Max value depth in m: " << maxVal << std::endl;
    // Count valid depth points
    cv::Mat valid_mask = depth > 0;
    int valid_points = cv::countNonZero(valid_mask);
    std::cout << "[DEPTH] Number of depth points > 0: " << valid_points << std::endl;

    // Normalize for visualization
    cv::Mat depth_vis;
    depth.convertTo(depth_vis, CV_32F, 1.0/(maxVal - minVal), -minVal/(maxVal - minVal));
    depth_vis.convertTo(depth_vis, CV_8UC1, 255.0);

    cv::imwrite(outputPath, depth_vis);

    return depth;

}



inline cv::Mat readPFM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open PFM file: " + filename);
    }

    // Read header
    std::string format;
    std::getline(file, format);
    if (format != "Pf") {  // Pf for grayscale, PF for RGB
        throw std::runtime_error("Invalid PFM format: " + format);
    }

    // Read dimensions
    int width, height;
    file >> width >> height;
    file.ignore(1); // Skip newline

    // Read scale/endianness
    float scale;
    file >> scale;
    file.ignore(1); // Skip newline
    bool bigEndian = (scale > 0.0f);
    std::cout << "Scale factor: " << scale << std::endl;
    // scale = std::abs(scale);

    // Create matrix to store the data
    cv::Mat data(height, width, CV_32FC1);

    // Read the data
    file.read(reinterpret_cast<char*>(data.data), width * height * sizeof(float));

    // Handle endianness if needed
    if (bigEndian != (htonl(1) == 1)) {
        for (int i = 0; i < height * width; i++) {
            float* pixel = &data.at<float>(i);
            char* bytes = reinterpret_cast<char*>(pixel);
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
        }
    }

    // After reading data and before scaling
    // Handle infinite values
    cv::Mat inf_mask = (data == std::numeric_limits<float>::infinity()) |
                            (data == -std::numeric_limits<float>::infinity());
    data.setTo(NAN, inf_mask);

    // data.setTo(NAN, cv::Mat(data == std::numeric_limits<float>::infinity()));
    // data.setTo(NAN, cv::Mat(data == -std::numeric_limits<float>::infinity()));
    // data.setTo(NAN, cv::Mat(data != data));  // Handle NaN

    // For PFM files, we typically want to divide by the scale factor when it's negative
    // (indicating little-endian) rather than multiply

    /*
    // Debug prints
    double min_val, max_val;
    cv::minMaxLoc(data, &min_val, &max_val);
    std::cout << "Data range before scaling - Min: " << min_val << ", Max: " << max_val << std::endl;

    // data = data * scale;  // Apply the scale factor

    // data = data * std::abs(scale);

    cv::minMaxLoc(data, &min_val, &max_val);
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Data range after scaling - Min: " << min_val << ", Max: " << max_val << std::endl;
    */

    // Flip the image vertically as PFM stores rows bottom-to-top
    cv::flip(data, data, 0);

    return data;
}