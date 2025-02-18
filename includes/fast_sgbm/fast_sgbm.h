#ifndef FAST_SGBM_H
#define FAST_SGBM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

// Cost type
using CostType = uint16_t;

/**
 * @brief FastSGBM: A simplified version of Semi-Global Matching (SGM).
 *
 * This implementation uses the Census transform for computing matching costs,
 * aggregates costs from multiple directions, and uses a Winner-Take-All strategy
 * to generate the final disparity map.
 */
class FastSGBM
{
public:
    /**
     * @brief Constructor.
     *
     * @param rows       Number of image rows.
     * @param cols       Number of image columns.
     * @param d_range    Disparity range (from 0 to d_range - 1).
     * @param P1         Penalty for disparity changes of 1.
     * @param P2         Penalty for larger disparity changes.
     * @param applyBlur  Whether to apply Gaussian blur to the images before processing.
     */
    FastSGBM(int rows, int cols, int d_range, int P1, int P2, bool applyBlur = false);

    /**
     * @brief Compute left-to-right disparity.
     *
     * @param leftGray   Left image in grayscale.
     * @param rightGray  Right image in grayscale.
     * @param dispOut    Output disparity map (CV_8UC1).
     */
    void compute_disp(const cv::Mat &leftGray,
                      const cv::Mat &rightGray,
                      cv::Mat &dispOut);

    /**
     * @brief Compute both left-to-right and right-to-left disparity.
     * 
     * @param leftGray   Left image in grayscale.
     * @param rightGray  Right image in grayscale.
     * @param dispLeft   Output left-to-right disparity map.
     * @param dispRight  Output right-to-left disparity map.
     */
    void compute_disp_both(const cv::Mat &leftGray,
                           const cv::Mat &rightGray,
                           cv::Mat &dispLeft,
                           cv::Mat &dispRight);

private:
    /**
     * @brief Apply a 3x3 Census transform to the input image.
     *
     * The output is an 8-bit image that encodes local structure.
     *
     * @param img     Input image.
     * @param census  Output Census-transformed image.
     */
    void censusTransform(const cv::Mat &img, cv::Mat &census) const;

    /**
     * @brief Compute the Hamming distance between two 8-bit values.
     *
     * @param a  First value.
     * @param b  Second value.
     * @return unsigned char  The Hamming distance.
     */
    inline unsigned char hammingDist8(unsigned char a, unsigned char b) const;

    /**
     * @brief Build the pixel-level cost volume based on Census transform values.
     *
     * The cost volume is defined over image rows, columns, and disparity levels.
     *
     * @param censusL  Census transform of the left image.
     * @param censusR  Census transform of the right image.
     */
    void buildCostVolume(const cv::Mat &censusL, const cv::Mat &censusR);

    /**
     * @brief Perform SGM cost aggregation in one specified direction.
     *
     * The direction is defined by the row (dr) and column (dc) increments.
     *
     * @param dr  Row increment.
     * @param dc  Column increment.
     */
    void aggregateOneDirection(int dr, int dc);

    /**
     * @brief Compute the aggregated cost at a pixel for a given direction.
     *
     * This function separates the aggregation logic for pixel (r, c).
     *
     * @param r           Row index of the pixel.
     * @param c           Column index of the pixel.
     * @param dr          Row increment for the aggregation direction.
     * @param dc          Column increment for the aggregation direction.
     * @param aggregator  Output cost vector for this pixel.
     */
    void computeAggregatorAt(int r, int c, int dr, int dc, std::vector<CostType> &aggregator);

    /**
     * @brief Select the best disparity using a Winner-Take-All approach.
     *
     * The disparity with the lowest aggregated cost is chosen.
     *
     * @param dispOut  Output disparity map.
     */
    void selectDisparity(cv::Mat &dispOut);

private:
    int rows_, cols_;       // Image dimensions.
    int disp_range_;        // Number of disparity levels.
    int P1_, P2_;           // Penalty parameters for SGM.
    bool blur_;             // Whether to apply Gaussian blur before processing.

    cv::Mat censusL_, censusR_;  // Census transform results for left and right images.

    // Raw pixel-level cost volume: cost_[r][c][d]
    std::vector<CostType> cost_;

    // Aggregated cost volume after SGM: sumCost_[r][c][d]
    std::vector<CostType> sumCost_;
};

#endif // FAST_SGBM_H
