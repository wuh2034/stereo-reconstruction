#ifndef ESSENTIAL_MATRIX_H
#define ESSENTIAL_MATRIX_H

#include <opencv2/opencv.hpp>
#include <vector>

/* ========== Helper functions for E decomposition ========== */

/**
 * @brief Check if a 3D point is in front of the camera
 * @param P      3x4 projection matrix
 * @param point3D 4x1 homogeneous coordinates
 */
bool isInFrontOfCamera(const cv::Mat& P, const cv::Mat& point3D)
{
    cv::Mat proj = P * point3D;  // 3x1
    double w = proj.at<double>(2,0);
    double w3 = point3D.at<double>(3,0);
    return (w > 0.0 && w3 != 0.0);
}

/**
 * @brief Given P1, P2 and corresponding pixel points p1, p2, perform triangulation to get a 3D point (4x1)
 */
cv::Mat triangulatePoint(const cv::Mat& P1, const cv::Mat& P2,
                         const cv::Point2f& p1, const cv::Point2f& p2)
{
    cv::Mat A(4,4, CV_64F);
    A.row(0) = p1.x * P1.row(2) - P1.row(0);
    A.row(1) = p1.y * P1.row(2) - P1.row(1);
    A.row(2) = p2.x * P2.row(2) - P2.row(0);
    A.row(3) = p2.y * P2.row(2) - P2.row(1);

    cv::SVD svd(A);
    cv::Mat X = svd.vt.row(3).t(); // 4x1
    double w = X.at<double>(3,0);
    if (std::fabs(w) < 1e-12) w = 1e-12;
    X /= w;
    return X;
}

/**
 * @brief Decompose the essential matrix E to find the best (R, t)
 *        Criterion: the solution that yields the largest number of triangulated points in front of both cameras
 * @param E      3x3
 * @param pts1   left image points
 * @param pts2   right image points
 * @param K1     left camera intrinsics
 * @param bestR  output R
 * @param bestT  output t
 */
void decomposeEssentialMatrix(const cv::Mat& E,
                              const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& K1,
                              cv::Mat& bestR,
                              cv::Mat& bestT)
{
    // 1) SVD(E)
    cv::SVD svd(E, cv::SVD::FULL_UV);
    cv::Mat U = svd.u, Vt = svd.vt;
    cv::Mat W = (cv::Mat_<double>(3,3) <<
                 0, -1, 0,
                 1,  0, 0,
                 0,  0, 1);

    // 2) Two possible R
    cv::Mat R1 = U * W  * Vt;
    cv::Mat R2 = U * W.t() * Vt;

    // 3) Two possible t
    cv::Mat t1 = U.col(2).clone();
    cv::Mat t2 = -U.col(2).clone();

    // Flip if determinant < 0
    if (cv::determinant(R1) < 0) R1 = -R1;
    if (cv::determinant(R2) < 0) R2 = -R2;

    std::vector<cv::Mat> R_candidates{ R1, R1, R2, R2 };
    std::vector<cv::Mat> t_candidates{ t1, t2, t1, t2 };

    // Construct the left camera projection matrix P1 = K1 * [I|0], 3x4
    cv::Mat P1 = (cv::Mat_<double>(3,4) << 
        K1.at<double>(0,0), 0,               K1.at<double>(0,2), 0,
        0,               K1.at<double>(1,1), K1.at<double>(1,2), 0,
        0,               0,               1,                 0
    );

    int maxInFront = -1;
    bestR = cv::Mat::eye(3,3, CV_64F);
    bestT = (cv::Mat_<double>(3,1) << 0,0,0);

    // Loop over the 4 (R,t) combinations
    for (size_t i = 0; i < R_candidates.size(); i++)
    {
        cv::Mat R_ = R_candidates[i];
        cv::Mat t_ = t_candidates[i];

        // Construct the right camera projection matrix P2 = K1 * [R|t]
        // (Assume same focal length, simplified)
        cv::Mat Rt = cv::Mat::eye(4,4, CV_64F);
        R_.copyTo(Rt(cv::Rect(0,0,3,3)));
        t_.copyTo(Rt(cv::Rect(3,0,1,3)));
        cv::Mat P2 = K1 * Rt(cv::Rect(0,0,4,3));

        // Count how many points are in front
        int inFront = 0;
        for (size_t k = 0; k < pts1.size(); k++)
        {
            cv::Mat X = triangulatePoint(P1, P2, pts1[k], pts2[k]);
            if (isInFrontOfCamera(P1, X) && isInFrontOfCamera(P2, X))
                inFront++;
        }

        if (inFront > maxInFront)
        {
            maxInFront = inFront;
            bestR = R_.clone();
            bestT = t_.clone();
        }
    }

    std::cout << "[decomposeEssentialMatrix] => best R:\n" << bestR 
              << "\nbest t:\n" << bestT
              << "\nInFrontCount=" << maxInFront << "/" << pts1.size() << std::endl;
}

#endif // ESSENTIAL_MATRIX_H
