#ifndef WARP_PERSPECTIVE_H
#define WARP_PERSPECTIVE_H

#include <opencv2/opencv.hpp>

// Custom implementation of warpPerspective (bilinear interpolation)
void myWarpPerspective(const cv::Mat &src, cv::Mat &dst, const cv::Mat &H, cv::Size dsize)
{
    // Compute inverse homography for backward mapping
    cv::Mat H_inv = H.inv();
    dst = cv::Mat::zeros(dsize, src.type());
    
    for (int y = 0; y < dsize.height; y++) {
        for (int x = 0; x < dsize.width; x++) {
            // Destination pixel (x, y, 1)
            cv::Mat p = (cv::Mat_<double>(3,1) << x, y, 1.0);
            cv::Mat p_src = H_inv * p;
            double w = p_src.at<double>(2,0);
            if (std::fabs(w) < 1e-10) continue;
            double u = p_src.at<double>(0,0) / w;
            double v = p_src.at<double>(1,0) / w;
            
            // If the computed source coordinate is within bounds, perform bilinear interpolation
            if (u >= 0 && u < src.cols - 1 && v >= 0 && v < src.rows - 1) {
                int x0 = static_cast<int>(floor(u));
                int y0 = static_cast<int>(floor(v));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                double a = u - x0;
                double b = v - y0;
                
                if (src.channels() == 1) {
                    double I00 = src.at<uchar>(y0, x0);
                    double I01 = src.at<uchar>(y0, x1);
                    double I10 = src.at<uchar>(y1, x0);
                    double I11 = src.at<uchar>(y1, x1);
                    double val = (1 - a) * (1 - b) * I00 + a * (1 - b) * I01 +
                                 (1 - a) * b * I10 + a * b * I11;
                    dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
                }
                else if (src.channels() == 3) {
                    cv::Vec3b I00 = src.at<cv::Vec3b>(y0, x0);
                    cv::Vec3b I01 = src.at<cv::Vec3b>(y0, x1);
                    cv::Vec3b I10 = src.at<cv::Vec3b>(y1, x0);
                    cv::Vec3b I11 = src.at<cv::Vec3b>(y1, x1);
                    cv::Vec3b val;
                    for (int c = 0; c < 3; c++) {
                        double interp = (1 - a) * (1 - b) * I00[c] +
                                        a * (1 - b) * I01[c] +
                                        (1 - a) * b * I10[c] +
                                        a * b * I11[c];
                        val[c] = cv::saturate_cast<uchar>(interp);
                    }
                    dst.at<cv::Vec3b>(y, x) = val;
                }
            }
        }
    }
}


#endif // WARP_PERSPECTIVE_H