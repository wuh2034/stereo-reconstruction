#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "includes/common.hpp"
#include "includes/depth_map_generation.h"
#include "includes/mesh_generation.h"
// double computeReprojectionError(
//     const std::vector<cv::Point2f> &pointsLeft,
//     const std::vector<cv::Point2f> &pointsRight,
//     const cv::Mat &F)
// {
//     if (pointsLeft.size() != pointsRight.size())
//     {
//         std::cerr << "[computeReprojectionError] Mismatch in points size!" << std::endl;
//         return -1.0;
//     }

//     std::vector<double> errors;
//     errors.reserve(pointsLeft.size());

//     for (size_t i = 0; i < pointsLeft.size(); ++i)
//     {
//         cv::Mat x1 = (cv::Mat_<double>(3, 1) << pointsLeft[i].x, pointsLeft[i].y, 1.0);
//         cv::Mat x2 = (cv::Mat_<double>(3, 1) << pointsRight[i].x, pointsRight[i].y, 1.0);

//         // The epipolar constraint error: | x2^T * F * x1 |
//         double error = std::abs(x2.dot(F * x1));
//         errors.push_back(error);
//     }

//     // Here we simply return the sum of squares
//     double sum_sq = 0.0;
//     for (auto &e : errors)
//         sum_sq += (e * e);

//     return sum_sq;
// }
// //Based on the Ransac to compute the F->the threshold for Shopvac dataset 
// cv::Mat processFeatureMethod(const std::string &method,
//                              std::vector<cv::Point2f> &pointsLeft,
//                              std::vector<cv::Point2f> &pointsRight,
//                              const std::vector<cv::DMatch> &good_matches)
// {

//     if (good_matches.empty())
//     {
//         std::cerr << "[ERROR] good_matches is empty." << std::endl;
//         return cv::Mat();
//     }

   
//     static const std::unordered_map<std::string, double> thresholdMap = {
//         {"ORB", 0.03},
//         {"SIFT", 0.018},
//         {"BRISK", 0.002}};

//     auto it = thresholdMap.find(method);
//     if (it == thresholdMap.end())
//     {
//         std::cerr << "[WARN] unsupported detective method: " << method << std::endl;
//         return cv::Mat();
//     }
//     double threshold = it->second;

//     // compute the fundamental matrix
//     int inlierCount = 0;
//     std::vector<cv::Point2f> ptsLeftCopy = pointsLeft;
//     std::vector<cv::Point2f> ptsRightCopy = pointsRight;
//     cv::Mat F = ransacFundamentalMatrix(ptsLeftCopy, ptsRightCopy, 3000, threshold, inlierCount);
//     if (F.empty())
//     {
//         std::cerr << "[WARN] " << method << ": F is empty." << std::endl;
//         return cv::Mat();
//     }

//     std::cout << "[INFO] " << method << " => #inliers = " << inlierCount << std::endl;

//     // calculate the reprojection error
//     double error_all = computeReprojectionError(pointsLeft, pointsRight, F);
//     std::cout << "[INFO] the error for ransac: " << error_all << std::endl;
//     double average_error = error_all / static_cast<double>(good_matches.size());
//     std::cout << "[INFO] the average error of each match: " << average_error << std::endl;

//     return F;
// }

// Compute disparity map using OpenCV's built-in StereoSGBM algorithm
cv::Mat computeDisparitySGBM(const cv::Mat &leftGray, const cv::Mat &rightGray)
{
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0,    // minDisparity
        128,  // numDisparities
        23    // blockSize
    );
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity);
    return disparity;
}

// Compute disparity map using a custom FastSGBM implementation
cv::Mat computeDisparityFastSGBM(cv::Mat &leftGray, cv::Mat &rightGray, int dispRange, int p1, int p2)
{
    FastSGBM sgbm(leftGray.rows, leftGray.cols, dispRange, p1, p2, true);
    cv::Mat disparity;
    sgbm.compute_disp(leftGray, rightGray, disparity);
    return disparity;
}

// Find the minimal bounding rectangle that encloses all non-black pixels.
static cv::Rect findNonBlackROI(const cv::Mat &img)
{
    int minx = img.cols, miny = img.rows;
    int maxx = -1, maxy = -1;
    for (int y = 0; y < img.rows; y++)
    {
        const uchar* rowPtr = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols; x++)
        {
            int b = rowPtr[3 * x + 0];
            int g = rowPtr[3 * x + 1];
            int r = rowPtr[3 * x + 2];
            if (b != 0 || g != 0 || r != 0)
            {
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
            }
        }
    }
    if (maxx < 0 || maxy < 0)
        return cv::Rect(0, 0, 1, 1);
    return cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
}

// Apply a homography transformation to an image and return the transformed corner points.
cv::Mat warpWithBoundingRect(const cv::Mat &img, const cv::Mat &H, std::vector<cv::Point2f> &cornersOut)
{
    std::vector<cv::Point2f> cornersIn = { 
        cv::Point2f(0.f, 0.f),
        cv::Point2f(static_cast<float>(img.cols), 0.f),
        cv::Point2f(static_cast<float>(img.cols), static_cast<float>(img.rows)),
        cv::Point2f(0.f, static_cast<float>(img.rows))
    };
    std::vector<cv::Point2f> cornersTrans;
    cv::perspectiveTransform(cornersIn, cornersTrans, H);
    cornersOut = cornersTrans;
    // Currently returns an empty image; warpPerspective can be adjusted later based on the bounding rectangle.
    return cv::Mat();
}

// Compute the union bounding rectangle of multiple sets of points and construct a translation matrix T (3x3)
// so that the top-left corner of the union rectangle is at (0, 0).
cv::Rect computeUnionBoundingRectAndShift(
    const std::vector<std::vector<cv::Point2f>> &allCorners,
    cv::Mat &T)
{
    float minX = 1e10f, minY = 1e10f;
    float maxX = -1e10f, maxY = -1e10f;
    for (auto &corners : allCorners)
    {
        for (auto &pt : corners)
        {
            if (pt.x < minX) minX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y > maxY) maxY = pt.y;
        }
    }
    int width  = static_cast<int>(std::ceil(maxX - minX));
    int height = static_cast<int>(std::ceil(maxY - minY));
    T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0,2) = -minX;
    T.at<double>(1,2) = -minY;
    return cv::Rect(0, 0, width, height);
}

int main()
{
    // Dataset configuration: choose "Shopvac_imperfect" or "artroom1"
    std::string dataset = "Shopvac_imperfect";
    // Automatically set image paths, camera intrinsics, focal length, and baseline based on the dataset
    std::string leftImagePath, rightImagePath;
    cv::Mat K0, K1;
    cv::Mat D0, D1;
    float focal_length, baseline;
    if(dataset == "Shopvac_imperfect")
    {
        leftImagePath = "../Datasets/Shopvac-imperfect/im0.png";
        rightImagePath = "../Datasets/Shopvac-imperfect/im1.png";
        K0 = (cv::Mat_<double>(3,3) << 7228.4, 0, 1112.085,
                                        0, 7228.4, 1010.431,
                                        0, 0, 1.0);
        K1 = (cv::Mat_<double>(3,3) << 7228.4, 0, 1628.613,
                                        0, 7228.4, 1010.431,
                                        0, 0, 1.0);
        focal_length = 7228.4;
        baseline = 379.965;
    }
    else if(dataset == "artroom1")
    {
        leftImagePath = "../Datasets/artroom1/im0.png";
        rightImagePath = "../Datasets/artroom1/im1.png";
        K0 = (cv::Mat_<double>(3,3) << 1733.74, 0, 792.27,
                                        0, 1733.74, 541.89,
                                        0, 0, 1.0);
        K1 = (cv::Mat_<double>(3,3) << 1733.74, 0, 792.27,
                                        0, 1733.74, 541.89,
                                        0, 0, 1.0);
        focal_length = 1733.74;
        baseline = 536.62;
    }
    // Set distortion parameters (all zeros in this case)
    D0 = cv::Mat::zeros(1,5,CV_64F);
    D1 = cv::Mat::zeros(1,5,CV_64F);

    // Parameters for depth map and mesh generation
    float upperDistanceThreshold = 50.0;
    float lowerDistanceThreshold = 0.1;

    // Load left and right images
    cv::Mat leftImage  = cv::imread(leftImagePath, cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "[Error] Failed to load images.\n";
        return -1;
    }

    // Define feature extraction methods along with their matching norms
    struct FeatureMethod {
        std::string name;
        cv::Ptr<cv::Feature2D> extractor;
        int normType;
    };
    std::vector<FeatureMethod> methods = {
        { "ORB",   cv::ORB::create(),   cv::NORM_HAMMING },
        { "SIFT",  cv::SIFT::create(),  cv::NORM_L2      },
        { "BRISK", cv::BRISK::create(), cv::NORM_HAMMING }
    };

    // Process each feature extraction method
    for (auto &method : methods) {
        std::cout << "\n========== " << method.name << " ==========\n";

        // Step (a): Compute keypoints and descriptors
        cv::Mat leftGray, rightGray;
        cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> kpLeft, kpRight;
        cv::Mat descLeft, descRight;
        method.extractor->detectAndCompute(leftGray, cv::noArray(), kpLeft, descLeft);
        method.extractor->detectAndCompute(rightGray, cv::noArray(), kpRight, descRight);
        if (descLeft.empty() || descRight.empty()) {
            std::cerr << "[WARN] " << method.name << ": no descriptors.\n";
            continue;
        }

        // Step (b1): KNN matching followed by a ratio test
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(method.normType, false);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(descLeft, descRight, knn, 2);
        std::vector<cv::DMatch> goodMatches;
        double ratioThresh = 0.75;
        for (auto &m : knn) {
            if (m.size() == 2 && m[0].distance < ratioThresh * m[1].distance)
                goodMatches.push_back(m[0]);
        }
        // Step (b2): dist threshold
        // for (int i = 0; i < descLeft.rows; i++)
        // {
        //     if (matches[i].distance <= std::max(3.5 * min_dist, 30.0))
        //     {
        //         goodMatches.push_back(matches[i]);
        //     }
        // }
        if (goodMatches.size() < 8) {
            std::cerr << "[WARN] " << method.name << ": Not enough good matches.\n";
            continue;
        }

        // Step (c): Collect matching points from both images
        std::vector<cv::Point2f> ptsLeft, ptsRight;
        for (auto &gm : goodMatches) {
            ptsLeft.push_back(kpLeft[gm.queryIdx].pt);
            ptsRight.push_back(kpRight[gm.trainIdx].pt);
        }

        // Step (d1): Estimate the fundamental matrix F using RANSAC
        int inlierCount = 0;
        cv::Mat F = ransacFundamentalMatrix(ptsLeft, ptsRight, 5000, 1.5f, inlierCount);
        if (F.empty()) {
            std::cerr << "[WARN] " << method.name << ": F is empty.\n";
            continue;
        }
        std::cout << "[INFO] " << method.name << " => #inliers=" << inlierCount << "\n";
        // Step (d2): specific threshold for the Shopvac dataset
        // cv::Mat F = ransacFundamentalMatrix(method.name,ptsLeft, ptsRight,goodMatches );

        // --- Step (1): Uncalibrated Rectification ---
        cv::Mat dispU8; // Disparity map from uncalibrated method
        {
            cv::Mat H1, H2;
            cv::stereoRectifyUncalibrated(ptsLeft, ptsRight, F, leftImage.size(), H1, H2);
            cv::Mat rectLeftU, rectRightU;
            cv::warpPerspective(leftImage, rectLeftU, H1, leftImage.size());
            cv::warpPerspective(rightImage, rectRightU, H2, leftImage.size());
            cv::Mat stereoUncalib;
            cv::hconcat(rectLeftU, rectRightU, stereoUncalib);
            // Draw horizontal lines on the uncalibrated rectified image for visualization
            for (int y = 50; y < stereoUncalib.rows; y += 100) {
                cv::line(stereoUncalib, cv::Point(0, y), cv::Point(stereoUncalib.cols-1, y),
                         cv::Scalar(0, 255, 0), 1);
            }
            std::string uncalibName = "uncalib_rectify_opencv_" + method.name + ".png";
            cv::imwrite(uncalibName, stereoUncalib);
            cv::Mat rectLeftUGray, rectRightUGray;
            cv::cvtColor(rectLeftU, rectLeftUGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRightU, rectRightUGray, cv::COLOR_BGR2GRAY);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
            clahe->apply(rectLeftUGray, rectLeftUGray);
            clahe->apply(rectRightUGray, rectRightUGray);
            cv::Mat dispU = computeDisparitySGBM(rectLeftUGray, rectRightUGray);
            double minVal, maxVal;
            cv::minMaxLoc(dispU, &minVal, &maxVal);
            dispU.convertTo(dispU8, CV_8U, 255.0/(maxVal - minVal + 1e-6));
            std::string dispUName = "uncalib_disparity_opencv_" + method.name + ".png";
            cv::imwrite(dispUName, dispU8);

            // For BRISK, generate the depth map and mesh from the computed disparity
            if (method.name == "BRISK") {
                std::string depth_map_nc = "/workspace/results/" + dataset + "_depth_map_uncalib_opencv_" + method.name + ".png";
                cv::Mat depth_map = computeDepthMapAndVis(dispU8, focal_length, baseline, depth_map_nc,
                                                           upperDistanceThreshold, lowerDistanceThreshold);
                std::string mesh_output_nc = "/workspace/results/" + dataset + "_mesh_uncalib_opencv_" + method.name + ".off";
                generateMeshFromDepth(depth_map, K0, dataset + "_uncalib_" + method.name,
                                      upperDistanceThreshold, lowerDistanceThreshold);
            }
        }

        // --- Step (2): Calibrated Rectification (Custom Implementation) ---
        cv::Mat dispC8; // Disparity map from custom calibrated method
        {
            cv::Mat E = K1.t() * F * K0;
            cv::Mat bestR, bestT;
            decomposeEssentialMatrix(E, ptsLeft, ptsRight, K0, bestR, bestT);
            cv::Mat P_left = cv::Mat::zeros(3,4,CV_64F);
            K0.copyTo(P_left(cv::Rect(0,0,3,3)));
            cv::Mat RT;
            cv::hconcat(bestR, bestT, RT);
            cv::Mat P_right = K1 * RT;
            cv::Mat H_left = P_left(cv::Rect(0,0,3,3)) * K0.inv();
            cv::Mat H_right = P_right(cv::Rect(0,0,3,3)) * K1.inv();
            cv::Mat rectLeftC, rectRightC;
            myWarpPerspective(leftImage, rectLeftC, H_left, leftImage.size());
            myWarpPerspective(rightImage, rectRightC, H_right, leftImage.size());
            cv::Mat stereoCalib;
            cv::hconcat(rectLeftC, rectRightC, stereoCalib);
            // Draw horizontal lines on the custom calibrated rectified image for visual inspection
            for (int y = 50; y < stereoCalib.rows; y += 100) {
                cv::line(stereoCalib, cv::Point(0, y), cv::Point(stereoCalib.cols-1, y),
                         cv::Scalar(0, 255, 0), 1);
            }
            std::string calibName = "ours_calib_rectify_" + method.name + ".png";
            cv::imwrite(calibName, stereoCalib);
            cv::Mat rectLeftCGray, rectRightCGray;
            cv::cvtColor(rectLeftC, rectLeftCGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRightC, rectRightCGray, cv::COLOR_BGR2GRAY);
            cv::Mat rectLeftCGraySmall, rectRightCGraySmall;
            cv::resize(rectLeftCGray, rectLeftCGraySmall, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            cv::resize(rectRightCGray, rectRightCGraySmall, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            int dispRangeCalib = 256, p1Calib = 10, p2Calib = 5;
            cv::Mat dispC = computeDisparityFastSGBM(rectLeftCGraySmall, rectRightCGraySmall, dispRangeCalib, p1Calib, p2Calib);
            double minValC, maxValC;
            cv::minMaxLoc(dispC, &minValC, &maxValC);
            dispC.convertTo(dispC8, CV_8U, 255.0/(maxValC - minValC + 1e-6));
            std::string dispCName = "ours_calib_disparity_" + method.name + ".png";
            cv::imwrite(dispCName, dispC8);

            // For BRISK, generate the depth map and mesh from the custom calibrated disparity
            if (method.name == "BRISK") {
                std::string depth_map_custom = "/workspace/results/" + dataset + "_depth_map_calib_ours_" + method.name + ".png";
                cv::Mat depth_map = computeDepthMapAndVis(dispC8, focal_length, baseline, depth_map_custom,
                                                           upperDistanceThreshold, lowerDistanceThreshold);
                std::string mesh_output_custom = "/workspace/results/" + dataset + "_mesh_calib_ours_" + method.name + ".off";
                generateMeshFromDepth(depth_map, K0, dataset + "_calib_custom_" + method.name,
                                      upperDistanceThreshold, lowerDistanceThreshold);
            }
        }

        // --- Step (3): Calibrated Rectification using OpenCV's Built-in Functions ---
        cv::Mat dispC8_op; // Disparity map from OpenCV's calibrated rectification
        {
            cv::Mat E = K1.t() * F * K0;
            cv::Mat bestR_op, bestT_op;
            decomposeEssentialMatrix(E, ptsLeft, ptsRight, K0, bestR_op, bestT_op);
            cv::Size imageSize(leftImage.cols, leftImage.rows);
            cv::Mat RR1, RR2, PP1, PP2, Q;
            cv::stereoRectify(K0, D0, K1, D1, imageSize,
                              bestR_op, bestT_op, RR1, RR2, PP1, PP2, Q,
                              0, -1, imageSize);
            cv::Mat mapxL, mapyL, mapxR, mapyR;
            cv::initUndistortRectifyMap(K0, D0, RR1, PP1, imageSize, CV_32FC1, mapxL, mapyL);
            cv::initUndistortRectifyMap(K1, D1, RR2, PP2, imageSize, CV_32FC1, mapxR, mapyR);
            cv::Mat rectLeftC_op, rectRightC_op;
            cv::remap(leftImage, rectLeftC_op, mapxL, mapyL, cv::INTER_LINEAR);
            cv::remap(rightImage, rectRightC_op, mapxR, mapyR, cv::INTER_LINEAR);
            cv::Mat stereoCalib_op;
            cv::hconcat(rectLeftC_op, rectRightC_op, stereoCalib_op);
            // Draw horizontal lines on the OpenCV calibrated rectified image for inspection
            for (int y = 50; y < stereoCalib_op.rows; y += 100) {
                cv::line(stereoCalib_op, cv::Point(0, y), cv::Point(stereoCalib_op.cols-1, y),
                         cv::Scalar(0, 255, 0), 1);
            }
            std::string calibName_op = "calib_opencv_" + method.name + ".png";
            cv::imwrite(calibName_op, stereoCalib_op);
            cv::Mat rectLeftCGray_op, rectRightCGray_op;
            cv::cvtColor(rectLeftC_op, rectLeftCGray_op, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRightC_op, rectRightCGray_op, cv::COLOR_BGR2GRAY);
            cv::Mat dispC_op = computeDisparitySGBM(rectLeftCGray_op, rectRightCGray_op);
            double minv_op, maxv_op;
            cv::minMaxLoc(dispC_op, &minv_op, &maxv_op);
            dispC_op.convertTo(dispC8_op, CV_8U, 255.0/(maxv_op - minv_op + 1e-6));
            std::string dispCName_op = "calib_disparity_opencv_" + method.name + ".png";
            cv::imwrite(dispCName_op, dispC8_op);

            // For BRISK, generate the depth map and mesh from the OpenCV calibrated disparity
            if (method.name == "BRISK") {
                std::string depth_map_op = "/workspace/results/" + dataset + "_depth_map_calib_opencv_" + method.name + ".png";
                cv::Mat depth_map = computeDepthMapAndVis(dispC8_op, focal_length, baseline, depth_map_op,
                                                           upperDistanceThreshold, lowerDistanceThreshold);
                std::string mesh_output_op = "/workspace/results/" + dataset + "_mesh_calib_opencv_" + method.name + ".off";
                generateMeshFromDepth(depth_map, K0, dataset + "_calib_opencv_" + method.name,
                                      upperDistanceThreshold, lowerDistanceThreshold);
            }
        }
    }

    return 0;
}
