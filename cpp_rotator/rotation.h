#ifndef ROTATION_H
#define ROTATION_H

#include <opencv2/opencv.hpp>

// Referenčné metódy (OpenCV)
cv::Mat rotate_nearest_ref(const cv::Mat& src, double angle_deg, bool cut_corners);
cv::Mat rotate_bilinear_ref(const cv::Mat& src, double angle_deg, bool cut_corners);
cv::Mat rotate_lanczos_ref(const cv::Mat& src, double angle_deg, bool cut_corners);

// Manuálne metódy (backward mapping s vlastnou interpoláciou)
cv::Mat rotate_nearest_manual(const cv::Mat& src, double angle_deg, bool cut_corners);
cv::Mat rotate_bilinear_manual(const cv::Mat& src, double angle_deg, bool cut_corners);
cv::Mat rotate_lanczos_manual(const cv::Mat& src, double angle_deg, bool cut_corners, int a);

#endif