#ifndef EPIPOLAR_GEOMETRY_H
#define EPIPOLAR_GEOMETRY_H

#include <opencv2/opencv.hpp>
#include <vector>

void normalizePoints(const std::vector<cv::Point2f> &points,
                     std::vector<cv::Point2f> &normalizedPoints,
                     cv::Mat &T)
{
    // 1) Compute centroid
    cv::Point2f center(0, 0);
    for (auto &p : points)
        center += p;
    center *= (1.0f / points.size());

    // 2) Compute average distance
    double avgDist = 0.0;
    for (auto &p : points)
        avgDist += cv::norm(p - center);
    avgDist /= points.size();

    // 3) Scale factor
    double scale = std::sqrt(2.0) / avgDist;

    // 4) Construct a 3x3 transform matrix
    T = (cv::Mat_<double>(3,3) << 
         scale,    0,   -scale*center.x,
            0,   scale, -scale*center.y,
            0,     0,       1);

    // 5) Normalize the points with T
    normalizedPoints.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Mat p = (cv::Mat_<double>(3,1) << points[i].x, points[i].y, 1.0);
        cv::Mat p_norm = T * p;
        double w = p_norm.at<double>(2);
        normalizedPoints[i].x = static_cast<float>(p_norm.at<double>(0) / w);
        normalizedPoints[i].y = static_cast<float>(p_norm.at<double>(1) / w);
    }
}

/**
 * @brief Estimate the fundamental matrix F using the 8-point method + normalization
 */
cv::Mat computeFundamentalMatrix(const std::vector<cv::Point2f> &ptsLeft,
                                 const std::vector<cv::Point2f> &ptsRight)
{
    if (ptsLeft.size() < 8) {
        std::cerr << "[computeFundamentalMatrix] Not enough points!\n";
        return cv::Mat();
    }

    // 1) Normalize
    std::vector<cv::Point2f> normLeft, normRight;
    cv::Mat T1, T2;
    normalizePoints(ptsLeft,  normLeft,  T1);
    normalizePoints(ptsRight, normRight, T2);

    // 2) Construct matrix A
    size_t N = normLeft.size();
    Eigen::MatrixXd A(N, 9);
    for (size_t i = 0; i < N; i++)
    {
        double x1 = normLeft[i].x;
        double y1 = normLeft[i].y;
        double x2 = normRight[i].x;
        double y2 = normRight[i].y;

        A(i,0) = x2*x1;   A(i,1) = x2*y1;   A(i,2) = x2;
        A(i,3) = y2*x1;   A(i,4) = y2*y1;   A(i,5) = y2;
        A(i,6) = x1;      A(i,7) = y1;      A(i,8) = 1.0;
    }

    // 3) SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd f = V.col(8); // last column

    // 4) Convert to OpenCV Mat
    cv::Mat F(3,3, CV_64F);
    for (int i = 0; i < 9; i++) {
        F.at<double>(i/3, i%3) = f(i);
    }

    // 5) Enforce rank=2
    cv::SVD svdF(F);
    cv::Mat U = svdF.u, W = svdF.w, Vt = svdF.vt;
    W.at<double>(2,0) = 0.0;
    F = U * cv::Mat::diag(W) * Vt;

    // 6) Denormalize
    F = T2.t() * F * T1;
    return F;
}

/**
 * @brief Compute epipolar distance
 */
double computeEpipolarError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    cv::Mat x1 = (cv::Mat_<double>(3,1) << p1.x, p1.y, 1.0);
    cv::Mat line = F * x1; // [A B C]^T
    double A = line.at<double>(0,0);
    double B = line.at<double>(1,0);
    double C = line.at<double>(2,0);
    double denom = std::sqrt(A*A + B*B);
    if (denom < 1e-12) return 1e12; // avoid div0
    double dist = std::fabs(A*p2.x + B*p2.y + C)/denom;
    return dist;
}

/**
 * @brief Simple RANSAC for F estimation
 */
cv::Mat ransacFundamentalMatrix(std::vector<cv::Point2f> &pts1,
                                std::vector<cv::Point2f> &pts2,
                                int maxIters, float threshold,
                                int &bestInlierCount)
{
    bestInlierCount = 0;
    cv::Mat bestF;
    srand(static_cast<unsigned>(time(0)));

    if (pts1.size() < 8) {
        std::cerr << "[ransacFundamentalMatrix] Not enough points.\n";
        return cv::Mat();
    }

    for (int i = 0; i < maxIters; i++)
    {
        // randomly sample 8 points
        std::vector<cv::Point2f> s1, s2;
        for (int j = 0; j < 8; j++) {
            int idx = rand() % pts1.size();
            s1.push_back(pts1[idx]);
            s2.push_back(pts2[idx]);
        }
        cv::Mat F = computeFundamentalMatrix(s1, s2);
        if (F.empty()) continue;

        // count inliers
        int count = 0;
        for (size_t k = 0; k < pts1.size(); k++)
        {
            double err = computeEpipolarError(pts1[k], pts2[k], F);
            if (err < threshold) count++;
        }

        if (count > bestInlierCount) {
            bestInlierCount = count;
            bestF = F.clone();
        }
    }

    return bestF;
}

#endif // EPIPOLAR_GEOMETRY_H
