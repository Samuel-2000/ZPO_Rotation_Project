#include "rotation.h"
#include <cmath>
#include <vector>
#include <algorithm>

// ---------- PSNR ----------
double getPSNR(const cv::Mat& I1, const cv::Mat& I2) {
    if (I1.size() != I2.size() || I1.type() != I2.type()) {
        return -1.0;
    }
    cv::Mat diff;
    cv::absdiff(I1, I2, diff);
    diff.convertTo(diff, CV_64F);
    cv::Mat sq = diff.mul(diff);
    cv::Scalar s = cv::sum(sq);
    double total_pixels = I1.total() * I1.channels();
    double mse = (s[0] + s[1] + s[2]) / total_pixels;
    if (mse <= 1e-12) return 1e9;
    double maxVal = 255.0;
    double psnr = 10.0 * log10((maxVal * maxVal) / mse);
    return psnr;
}

// ---------- Interpolation helpers ----------
cv::Vec3b bilinear_interpolate(const cv::Mat& img, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;

    auto get_pixel = [&](int ix, int iy) -> cv::Vec3f {
        if (ix >= 0 && ix < img.cols && iy >= 0 && iy < img.rows)
            return cv::Vec3f(img.at<cv::Vec3b>(iy, ix));
        else
            return cv::Vec3f(0,0,0);
    };

    cv::Vec3f p00 = get_pixel(x0, y0);
    cv::Vec3f p10 = get_pixel(x1, y0);
    cv::Vec3f p01 = get_pixel(x0, y1);
    cv::Vec3f p11 = get_pixel(x1, y1);

    cv::Vec3f top = p00 * (1 - dx) + p10 * dx;
    cv::Vec3f bottom = p01 * (1 - dx) + p11 * dx;
    cv::Vec3f result = top * (1 - dy) + bottom * dy;

    return cv::Vec3b(cv::saturate_cast<uchar>(result[0]),
                     cv::saturate_cast<uchar>(result[1]),
                     cv::saturate_cast<uchar>(result[2]));
}

// ---------- Bicubic interpolation ----------
static double cubic_weight(double x) {
    // Mitchell‑Netravali / Catmull‑Rom kernel (standard bicubic)
    x = fabs(x);
    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    } else if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

cv::Vec3b bicubic_interpolate(const cv::Mat& img, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    float dx = x - x0;
    float dy = y - y0;

    cv::Vec3d accum(0.0, 0.0, 0.0);
    double weight_sum = 0.0;

    for (int m = -1; m <= 2; ++m) {
        int iy = y0 + m;
        if (iy < 0 || iy >= img.rows) continue;
        double wy = cubic_weight(m - dy);
        for (int n = -1; n <= 2; ++n) {
            int ix = x0 + n;
            if (ix < 0 || ix >= img.cols) continue;
            double wx = cubic_weight(n - dx);
            double w = wx * wy;
            if (w != 0.0) {
                cv::Vec3b p = img.at<cv::Vec3b>(iy, ix);
                accum[0] += w * p[0];
                accum[1] += w * p[1];
                accum[2] += w * p[2];
                weight_sum += w;
            }
        }
    }

    if (weight_sum > 1e-6) {
        accum /= weight_sum;
        return cv::Vec3b(cv::saturate_cast<uchar>(accum[0]),
                         cv::saturate_cast<uchar>(accum[1]),
                         cv::saturate_cast<uchar>(accum[2]));
    } else {
        return cv::Vec3b(0, 0, 0);
    }
}

static double lanczos_kernel(double x, int a) {
    if (x == 0.0) return 1.0;
    if (fabs(x) >= a) return 0.0;
    double pix = CV_PI * x;
    return a * sin(pix) * sin(pix / a) / (pix * pix);
}

cv::Vec3b lanczos_interpolate(const cv::Mat& img, float x, float y, int a) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    double sum_weight = 0.0;
    cv::Vec3d accum(0,0,0);

    for (int dy = -a + 1; dy <= a; ++dy) {
        for (int dx = -a + 1; dx <= a; ++dx) {
            int ix = x0 + dx;
            int iy = y0 + dy;
            double wx = lanczos_kernel(x - ix, a);
            double wy = lanczos_kernel(y - iy, a);
            double w = wx * wy;
            if (w != 0.0) {
                if (ix >= 0 && ix < img.cols && iy >= 0 && iy < img.rows) {
                    cv::Vec3b pixel = img.at<cv::Vec3b>(iy, ix);
                    accum[0] += w * pixel[0];
                    accum[1] += w * pixel[1];
                    accum[2] += w * pixel[2];
                }
                sum_weight += w;
            }
        }
    }

    if (sum_weight > 1e-6) {
        accum /= sum_weight;
        return cv::Vec3b(cv::saturate_cast<uchar>(accum[0]),
                         cv::saturate_cast<uchar>(accum[1]),
                         cv::saturate_cast<uchar>(accum[2]));
    } else {
        return cv::Vec3b(0,0,0);
    }
}

// ---------- compute bbox ----------
void compute_bbox(const cv::Mat& src, double angle_deg, cv::Size& dst_size, cv::Point2d& translation) {
    double rad = angle_deg * CV_PI / 180.0;
    double cos_a = cos(rad);
    double sin_a = sin(rad);
    double cx = src.cols / 2.0;
    double cy = src.rows / 2.0;

    cv::Mat T_origin = (cv::Mat_<double>(3,3) << 1,0,-cx, 0,1,-cy, 0,0,1);
    cv::Mat R = (cv::Mat_<double>(3,3) << cos_a, -sin_a, 0, sin_a, cos_a, 0, 0,0,1);
    cv::Mat T_back = (cv::Mat_<double>(3,3) << 1,0,cx, 0,1,cy, 0,0,1);
    cv::Mat M = T_back * R * T_origin;

    std::vector<cv::Point2d> corners = {
        cv::Point2d(0,0), cv::Point2d(src.cols,0), cv::Point2d(0,src.rows), cv::Point2d(src.cols,src.rows)
    };
    std::vector<cv::Point2d> rotated(4);
    for (int i=0; i<4; ++i) {
        rotated[i].x = M.at<double>(0,0)*corners[i].x + M.at<double>(0,1)*corners[i].y + M.at<double>(0,2);
        rotated[i].y = M.at<double>(1,0)*corners[i].x + M.at<double>(1,1)*corners[i].y + M.at<double>(1,2);
    }
    double min_x = rotated[0].x, max_x = rotated[0].x;
    double min_y = rotated[0].y, max_y = rotated[0].y;
    for (int i=1; i<4; ++i) {
        min_x = std::min(min_x, rotated[i].x);
        max_x = std::max(max_x, rotated[i].x);
        min_y = std::min(min_y, rotated[i].y);
        max_y = std::max(max_y, rotated[i].y);
    }
    dst_size.width  = cvRound(max_x - min_x);
    dst_size.height = cvRound(max_y - min_y);
    translation = cv::Point2d(-min_x, -min_y);
}

// ---------- reference methods (OpenCV warpAffine) ----------
cv::Mat rotate_nearest_ref(const cv::Mat& src, double angle_deg, bool cut_corners) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle_deg, 1.0);
    cv::Size dst_size;
    cv::Point2f translation(0,0);
    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        dst_size = full_size;
        translation = cv::Point2f(trans.x, trans.y);
        rot_mat.at<double>(0,2) += translation.x;
        rot_mat.at<double>(1,2) += translation.y;
    } else {
        dst_size = src.size();
    }
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, dst_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return dst;
}

cv::Mat rotate_bilinear_ref(const cv::Mat& src, double angle_deg, bool cut_corners) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle_deg, 1.0);
    cv::Size dst_size;
    cv::Point2f translation(0,0);
    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        dst_size = full_size;
        translation = cv::Point2f(trans.x, trans.y);
        rot_mat.at<double>(0,2) += translation.x;
        rot_mat.at<double>(1,2) += translation.y;
    } else {
        dst_size = src.size();
    }
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, dst_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return dst;
}

cv::Mat rotate_bicubic_ref(const cv::Mat& src, double angle_deg, bool cut_corners) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle_deg, 1.0);
    cv::Size dst_size;
    cv::Point2f translation(0, 0);
    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        dst_size = full_size;
        translation = cv::Point2f(trans.x, trans.y);
        rot_mat.at<double>(0, 2) += translation.x;
        rot_mat.at<double>(1, 2) += translation.y;
    } else {
        dst_size = src.size();
    }
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, dst_size, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst;
}

cv::Mat rotate_lanczos_ref(const cv::Mat& src, double angle_deg, bool cut_corners) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle_deg, 1.0);
    cv::Size dst_size;
    cv::Point2f translation(0,0);
    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        dst_size = full_size;
        translation = cv::Point2f(trans.x, trans.y);
        rot_mat.at<double>(0,2) += translation.x;
        rot_mat.at<double>(1,2) += translation.y;
    } else {
        dst_size = src.size();
    }
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, dst_size, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return dst;
}

// ---------- manual methods (backward mapping) ----------
cv::Mat rotate_nearest_manual(const cv::Mat& src, double angle_deg, bool cut_corners) {
    double rad = angle_deg * CV_PI / 180.0;
    double cx = src.cols / 2.0;
    double cy = src.rows / 2.0;

    cv::Mat T_origin = (cv::Mat_<double>(3,3) << 1,0,-cx, 0,1,-cy, 0,0,1);
    cv::Mat R = (cv::Mat_<double>(3,3) << cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0,0,1);
    cv::Mat T_back = (cv::Mat_<double>(3,3) << 1,0,cx, 0,1,cy, 0,0,1);
    cv::Mat M = T_back * R * T_origin;
    cv::Mat invM = M.inv(cv::DECOMP_SVD);

    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        cv::Mat dst = cv::Mat::zeros(full_size.height, full_size.width, CV_8UC3);
        for (int y = 0; y < full_size.height; ++y) {
            for (int x = 0; x < full_size.width; ++x) {
                double world_x = x - trans.x;
                double world_y = y - trans.y;
                double x_src = invM.at<double>(0,0)*world_x + invM.at<double>(0,1)*world_y + invM.at<double>(0,2);
                double y_src = invM.at<double>(1,0)*world_x + invM.at<double>(1,1)*world_y + invM.at<double>(1,2);
                int ix = cvRound(x_src);
                int iy = cvRound(y_src);
                if (ix >= 0 && ix < src.cols && iy >= 0 && iy < src.rows)
                    dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(iy, ix);
            }
        }
        return dst;
    } else {
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                double x_src = invM.at<double>(0,0)*x + invM.at<double>(0,1)*y + invM.at<double>(0,2);
                double y_src = invM.at<double>(1,0)*x + invM.at<double>(1,1)*y + invM.at<double>(1,2);
                int ix = cvRound(x_src);
                int iy = cvRound(y_src);
                if (ix >= 0 && ix < src.cols && iy >= 0 && iy < src.rows)
                    dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(iy, ix);
            }
        }
        return dst;
    }
}

cv::Mat rotate_bilinear_manual(const cv::Mat& src, double angle_deg, bool cut_corners) {
    double rad = angle_deg * CV_PI / 180.0;
    double cx = src.cols / 2.0;
    double cy = src.rows / 2.0;

    cv::Mat T_origin = (cv::Mat_<double>(3,3) << 1,0,-cx, 0,1,-cy, 0,0,1);
    cv::Mat R = (cv::Mat_<double>(3,3) << cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0,0,1);
    cv::Mat T_back = (cv::Mat_<double>(3,3) << 1,0,cx, 0,1,cy, 0,0,1);
    cv::Mat M = T_back * R * T_origin;
    cv::Mat invM = M.inv(cv::DECOMP_SVD);

    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        cv::Mat dst = cv::Mat::zeros(full_size.height, full_size.width, CV_8UC3);
        for (int y = 0; y < full_size.height; ++y) {
            for (int x = 0; x < full_size.width; ++x) {
                double world_x = x - trans.x;
                double world_y = y - trans.y;
                double x_src = invM.at<double>(0,0)*world_x + invM.at<double>(0,1)*world_y + invM.at<double>(0,2);
                double y_src = invM.at<double>(1,0)*world_x + invM.at<double>(1,1)*world_y + invM.at<double>(1,2);
                dst.at<cv::Vec3b>(y, x) = bilinear_interpolate(src, (float)x_src, (float)y_src);
            }
        }
        return dst;
    } else {
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                double x_src = invM.at<double>(0,0)*x + invM.at<double>(0,1)*y + invM.at<double>(0,2);
                double y_src = invM.at<double>(1,0)*x + invM.at<double>(1,1)*y + invM.at<double>(1,2);
                dst.at<cv::Vec3b>(y, x) = bilinear_interpolate(src, (float)x_src, (float)y_src);
            }
        }
        return dst;
    }
}


// ---------- Bicubic manual ----------
cv::Mat rotate_bicubic_manual(const cv::Mat& src, double angle_deg, bool cut_corners) {
    double rad = angle_deg * CV_PI / 180.0;
    double cx = src.cols / 2.0;
    double cy = src.rows / 2.0;

    cv::Mat T_origin = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
    cv::Mat R = (cv::Mat_<double>(3, 3) << cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0, 0, 1);
    cv::Mat T_back = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
    cv::Mat M = T_back * R * T_origin;
    cv::Mat invM = M.inv(cv::DECOMP_SVD);

    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        cv::Mat dst = cv::Mat::zeros(full_size.height, full_size.width, CV_8UC3);

        // Optional parallelisation (like Lanczos) – uncomment if needed
        // cv::parallel_for_(cv::Range(0, full_size.height), [&](const cv::Range& range) {
        //     for (int y = range.start; y < range.end; ++y) {
        for (int y = 0; y < full_size.height; ++y) {
            for (int x = 0; x < full_size.width; ++x) {
                double world_x = x - trans.x;
                double world_y = y - trans.y;
                double x_src = invM.at<double>(0, 0) * world_x + invM.at<double>(0, 1) * world_y + invM.at<double>(0, 2);
                double y_src = invM.at<double>(1, 0) * world_x + invM.at<double>(1, 1) * world_y + invM.at<double>(1, 2);
                dst.at<cv::Vec3b>(y, x) = bicubic_interpolate(src, (float)x_src, (float)y_src);
            }
        }
        // });
        return dst;
    } else {
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                double x_src = invM.at<double>(0, 0) * x + invM.at<double>(0, 1) * y + invM.at<double>(0, 2);
                double y_src = invM.at<double>(1, 0) * x + invM.at<double>(1, 1) * y + invM.at<double>(1, 2);
                dst.at<cv::Vec3b>(y, x) = bicubic_interpolate(src, (float)x_src, (float)y_src);
            }
        }
        return dst;
    }
}

cv::Mat rotate_lanczos_manual(const cv::Mat& src, double angle_deg, bool cut_corners, int a) {
    double rad = angle_deg * CV_PI / 180.0;
    double cx = src.cols / 2.0;
    double cy = src.rows / 2.0;

    cv::Mat T_origin = (cv::Mat_<double>(3,3) << 1,0,-cx, 0,1,-cy, 0,0,1);
    cv::Mat R = (cv::Mat_<double>(3,3) << cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0,0,1);
    cv::Mat T_back = (cv::Mat_<double>(3,3) << 1,0,cx, 0,1,cy, 0,0,1);
    cv::Mat M = T_back * R * T_origin;
    cv::Mat invM = M.inv(cv::DECOMP_SVD);

    if (!cut_corners) {
        cv::Size full_size;
        cv::Point2d trans;
        compute_bbox(src, angle_deg, full_size, trans);
        cv::Mat dst = cv::Mat::zeros(full_size.height, full_size.width, CV_8UC3);

        cv::parallel_for_(cv::Range(0, full_size.height), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < full_size.width; ++x) {
                    double world_x = x - trans.x;
                    double world_y = y - trans.y;
                    double x_src = invM.at<double>(0,0)*world_x + invM.at<double>(0,1)*world_y + invM.at<double>(0,2);
                    double y_src = invM.at<double>(1,0)*world_x + invM.at<double>(1,1)*world_y + invM.at<double>(1,2);
                    dst.at<cv::Vec3b>(y, x) = lanczos_interpolate(src, (float)x_src, (float)y_src, a);
                }
            }
        });
        return dst;
    } else {
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);

        cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < src.cols; ++x) {
                    double x_src = invM.at<double>(0,0)*x + invM.at<double>(0,1)*y + invM.at<double>(0,2);
                    double y_src = invM.at<double>(1,0)*x + invM.at<double>(1,1)*y + invM.at<double>(1,2);
                    dst.at<cv::Vec3b>(y, x) = lanczos_interpolate(src, (float)x_src, (float)y_src, a);
                }
            }
        });
        return dst;
    }
}


// ---------- maximal inner rectangle ----------
// compute maximal axis-aligned rectangle (width,height) that fits inside rotated rectangle
// Input w,h are original image width and height, angle_deg is rotation angle in degrees
void get_max_inner_rect(double w, double h, double angle_deg, double &out_w, double &out_h)
{
    // Normalize angle
    double angle = fmod(angle_deg, 180.0);
    if (angle < 0) angle += 180.0;
    if (angle > 90.0) angle = 180.0 - angle;

    double theta = angle * CV_PI / 180.0;

    double sin_a = fabs(sin(theta));
    double cos_a = fabs(cos(theta));

    if (sin_a == 0.0)
    {
        out_w = w;
        out_h = h;
        return;
    }

    if (cos_a == 0.0)
    {
        out_w = h;
        out_h = w;
        return;
    }

    bool width_longer = w >= h;

    double side_long  = width_longer ? w : h;
    double side_short = width_longer ? h : w;

    double wr, hr;

    if (side_short <= 2.0 * sin_a * cos_a * side_long)
    {
        double x = 0.5 * side_short;

        if (width_longer)
        {
            wr = x / sin_a;
            hr = x / cos_a;
        }
        else
        {
            wr = x / cos_a;
            hr = x / sin_a;
        }
    }
    else
    {
        double cos_2a = cos_a * cos_a - sin_a * sin_a;

        wr = (w * cos_a - h * sin_a) / cos_2a;
        hr = (h * cos_a - w * sin_a) / cos_2a;
    }

    out_w = fabs(wr);
    out_h = fabs(hr);
}