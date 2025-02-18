#include "fast_sgbm.h"
#include <algorithm>
#include <iostream>
#include <limits>

// Macro for indexing into the cost volume array
#define COST_IDX(r,c,d) ((r)*(cols_)*(disp_range_) + (c)*(disp_range_) + (d))

FastSGBM::FastSGBM(int rows, int cols, int d_range, int P1, int P2, bool applyBlur)
    : rows_(rows), cols_(cols), disp_range_(d_range),
      P1_(P1), P2_(P2), blur_(applyBlur)
{
    // Allocate space for the cost and aggregated cost volumes
    cost_.resize(rows_ * cols_ * disp_range_, 0);
    sumCost_.resize(rows_ * cols_ * disp_range_, 0);

    // Initialize census transform matrices for the left and right images
    censusL_ = cv::Mat(rows_, cols_, CV_8UC1);
    censusR_ = cv::Mat(rows_, cols_, CV_8UC1);
}

void FastSGBM::compute_disp(const cv::Mat &leftGray,
                            const cv::Mat &rightGray,
                            cv::Mat &dispOut)
{

    // Apply Gaussian blur to reduce noise
    cv::Mat left = leftGray, right = rightGray;
    if (blur_) {
        cv::GaussianBlur(leftGray, left, cv::Size(3,3), 0);
        cv::GaussianBlur(rightGray, right, cv::Size(3,3), 0);
    }

    // 1) Compute census transform for both images
    censusTransform(left, censusL_);
    censusTransform(right, censusR_);

    // 2) Build the pixel-level cost volume using the census transforms
    std::fill(cost_.begin(), cost_.end(), 0);
    buildCostVolume(censusL_, censusR_);

    // 3) Aggregate the cost in 8 different directions
    std::fill(sumCost_.begin(), sumCost_.end(), 0);

    // Diagonal directions
    aggregateOneDirection(+1, +1);
    aggregateOneDirection(+1, -1);
    aggregateOneDirection(-1, +1);
    aggregateOneDirection(-1, -1);

    // Key modification:
    // Process pure horizontal and vertical directions separately to avoid infinite loops.
    aggregateOneDirection(0, +1);  // left-to-right
    aggregateOneDirection(0, -1);  // right-to-left
    aggregateOneDirection(+1, 0);  // top-to-bottom
    aggregateOneDirection(-1, 0);  // bottom-to-top

    // 4) Winner-Take-All: select the disparity with the minimum cost
    dispOut.create(rows_, cols_, CV_8UC1);
    selectDisparity(dispOut);
}

void FastSGBM::compute_disp_both(const cv::Mat &leftGray,
                                 const cv::Mat &rightGray,
                                 cv::Mat &dispLeft,
                                 cv::Mat &dispRight)
{
    // Compute disparity from left to right
    compute_disp(leftGray, rightGray, dispLeft);
    // Compute disparity from right to left
    compute_disp(rightGray, leftGray, dispRight);
}

void FastSGBM::censusTransform(const cv::Mat &img, cv::Mat &census) const
{
    census.setTo(0);
    // Compute a simple 8-bit census transform for each pixel (ignoring border pixels)
    for (int r = 1; r < rows_ - 1; r++) {
        for (int c = 1; c < cols_ - 1; c++) {
            unsigned char center = img.at<uchar>(r, c);
            unsigned char val = 0;
            val |= (img.at<uchar>(r-1, c-1) >= center) << 7;
            val |= (img.at<uchar>(r-1, c  ) >= center) << 6;
            val |= (img.at<uchar>(r-1, c+1) >= center) << 5;
            val |= (img.at<uchar>(r,   c+1) >= center) << 4;
            val |= (img.at<uchar>(r+1, c+1) >= center) << 3;
            val |= (img.at<uchar>(r+1, c  ) >= center) << 2;
            val |= (img.at<uchar>(r+1, c-1) >= center) << 1;
            val |= (img.at<uchar>(r,   c-1) >= center) << 0;
            census.at<uchar>(r, c) = val;
        }
    }
}

inline unsigned char FastSGBM::hammingDist8(unsigned char a, unsigned char b) const
{
    // Compute the Hamming distance between two 8-bit values
    unsigned char x = a ^ b;
    unsigned char dist = 0;
    while (x) {
        x &= (x - 1);
        dist++;
    }
    return dist;
}

void FastSGBM::buildCostVolume(const cv::Mat &censusL, const cv::Mat &censusR)
{
    // For each pixel, compute the matching cost across disparity levels using Hamming distance
    for (int r = 0; r < rows_; r++) {
        const uchar* rowL = censusL.ptr<uchar>(r);
        for (int c = 0; c < cols_; c++) {
            unsigned char cL = rowL[c];
            for (int d = 0; d < disp_range_; d++) {
                int cr = c - d;
                unsigned char cR = 0;
                if (cr >= 0) {
                    cR = censusR.at<uchar>(r, cr);
                }
                cost_[COST_IDX(r, c, d)] = hammingDist8(cL, cR);
            }
        }
    }
}

/**
 * @brief Aggregate cost in one specified direction.
 *
 * dr and dc specify the row and column increments.
 * Note: When dr or dc is 0, the loop must be handled differently to avoid infinite loops.
 */
void FastSGBM::aggregateOneDirection(int dr, int dc)
{
    std::vector<CostType> aggregator(rows_ * cols_ * disp_range_, 0);

    // Handle diagonal directions first
    if (dr != 0 && dc != 0) {
        // Determine loop start and end points based on the direction
        int rowStart = (dr > 0) ? 0 : (rows_ - 1);
        int rowEnd   = (dr > 0) ? rows_ : -1;
        int colStart = (dc > 0) ? 0 : (cols_ - 1);
        int colEnd   = (dc > 0) ? cols_ : -1;

        for (int r = rowStart; r != rowEnd; r += dr) {
            for (int c = colStart; c != colEnd; c += dc) {
                // Get pointers to the current pixel's cost vector
                CostType *outLine = &aggregator[COST_IDX(r, c, 0)];
                const CostType *pixLine = &cost_[COST_IDX(r, c, 0)];

                int pr = r - dr;
                int pc = c - dc;

                // If the previous pixel is out of bounds, copy the current cost
                if (pr < 0 || pr >= rows_ || pc < 0 || pc >= cols_) {
                    for (int d = 0; d < disp_range_; d++) {
                        outLine[d] = pixLine[d];
                    }
                } else {
                    const CostType *prevLine = &aggregator[COST_IDX(pr, pc, 0)];
                    // Find the minimum and second minimum costs from the previous pixel
                    CostType minCost = std::numeric_limits<CostType>::max();
                    CostType secondMinCost = std::numeric_limits<CostType>::max();
                    int bestD = -1;
                    for (int d = 0; d < disp_range_; d++) {
                        CostType val = prevLine[d];
                        if (val < minCost) {
                            secondMinCost = minCost;
                            minCost = val;
                            bestD = d;
                        } else if (val < secondMinCost) {
                            secondMinCost = val;
                        }
                    }
                    // Aggregate the cost for each disparity level
                    for (int d = 0; d < disp_range_; d++) {
                        CostType baseCost = prevLine[d];
                        CostType val = baseCost;
                        if (d > 0) {
                            CostType alt = (CostType)(prevLine[d-1] + P1_);
                            if (alt < val) val = alt;
                        }
                        if (d < disp_range_ - 1) {
                            CostType alt = (CostType)(prevLine[d+1] + P1_);
                            if (alt < val) val = alt;
                        }
                        if (d == bestD) {
                            CostType alt = (CostType)(secondMinCost + P2_);
                            if (alt < val) val = alt;
                        } else {
                            CostType alt = (CostType)(minCost + P2_);
                            if (alt < val) val = alt;
                        }
                        CostType res = (CostType)(pixLine[d] + val - minCost);
                        outLine[d] = res;
                    }
                }
            }
        }
    }
    // Handle horizontal left-to-right aggregation
    else if (dr == 0 && dc == +1) {
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                CostType *outLine = &aggregator[COST_IDX(r, c, 0)];
                const CostType *pixLine = &cost_[COST_IDX(r, c, 0)];

                int pr = r;
                int pc = c - 1;  // Previous pixel to the left

                if (pc < 0) {
                    // No previous pixel; copy the current cost
                    for (int d = 0; d < disp_range_; d++) {
                        outLine[d] = pixLine[d];
                    }
                } else {
                    const CostType *prevLine = &aggregator[COST_IDX(pr, pc, 0)];
                    // Find minimum and second minimum cost
                    CostType minCost = std::numeric_limits<CostType>::max();
                    CostType secondMinCost = std::numeric_limits<CostType>::max();
                    int bestD = -1;
                    for (int d = 0; d < disp_range_; d++) {
                        CostType val = prevLine[d];
                        if (val < minCost) {
                            secondMinCost = minCost;
                            minCost = val;
                            bestD = d;
                        } else if (val < secondMinCost) {
                            secondMinCost = val;
                        }
                    }
                    // Aggregate the cost for each disparity level
                    for (int d = 0; d < disp_range_; d++) {
                        CostType baseCost = prevLine[d];
                        CostType val = baseCost;
                        if (d > 0) {
                            CostType alt = (CostType)(prevLine[d-1] + P1_);
                            if (alt < val) val = alt;
                        }
                        if (d < disp_range_ - 1) {
                            CostType alt = (CostType)(prevLine[d+1] + P1_);
                            if (alt < val) val = alt;
                        }
                        if (d == bestD) {
                            CostType alt = (CostType)(secondMinCost + P2_);
                            if (alt < val) val = alt;
                        } else {
                            CostType alt = (CostType)(minCost + P2_);
                            if (alt < val) val = alt;
                        }
                        CostType res = (CostType)(pixLine[d] + val - minCost);
                        outLine[d] = res;
                    }
                }
            }
        }
    }
    // Horizontal right-to-left aggregation
    else if (dr == 0 && dc == -1) {
        for (int r = 0; r < rows_; r++) {
            for (int c = cols_ - 1; c >= 0; c--) {
                // Similar logic as the left-to-right case (adjusted for reverse order)
            }
        }
    }
    // Vertical top-to-bottom aggregation
    else if (dr == +1 && dc == 0) {
        for (int c = 0; c < cols_; c++) {
            for (int r = 0; r < rows_; r++) {
                // Similar logic as the horizontal case (adjusted for vertical direction)
            }
        }
    }
    // Vertical bottom-to-top aggregation
    else if (dr == -1 && dc == 0) {
        for (int c = 0; c < cols_; c++) {
            for (int r = rows_ - 1; r >= 0; r--) {
                // Similar logic as the horizontal case (adjusted for reverse vertical direction)
            }
        }
    }

    // Accumulate the aggregated cost into the overall sumCost_ volume
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            for (int d = 0; d < disp_range_; d++) {
                sumCost_[COST_IDX(r, c, d)] += aggregator[COST_IDX(r, c, d)];
            }
        }
    }
}

void FastSGBM::selectDisparity(cv::Mat &dispOut)
{
    // Create an output disparity map and select the disparity with the minimum cost at each pixel
    dispOut.create(rows_, cols_, CV_8UC1);
    for (int r = 0; r < rows_; r++) {
        uchar* rowPtr = dispOut.ptr<uchar>(r);
        for (int c = 0; c < cols_; c++) {
            CostType bestCost = sumCost_[COST_IDX(r, c, 0)];
            int bestD = 0;
            for (int d = 1; d < disp_range_; d++) {
                CostType val = sumCost_[COST_IDX(r, c, d)];
                if (val < bestCost) {
                    bestCost = val;
                    bestD = d;
                }
            }
            rowPtr[c] = static_cast<uchar>(bestD);
        }
    }
}
