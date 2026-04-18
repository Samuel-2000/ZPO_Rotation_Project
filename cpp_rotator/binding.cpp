#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "rotation.h"

namespace py = pybind11;

cv::Mat numpy_to_cv(py::array_t<unsigned char>& input) {
    auto buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    return mat;
}

py::array_t<unsigned char> cv_to_numpy(const cv::Mat& mat) {
    py::array_t<unsigned char> result({ mat.rows, mat.cols, 3 });
    auto buf = result.request();
    std::memcpy(buf.ptr, mat.data, mat.total() * mat.elemSize());
    return result;
}

PYBIND11_MODULE(rotator_cpp, m) {
    m.doc() = "Image rotation module (C++ backend)";

    m.def("psnr", [](py::array_t<unsigned char> img1, py::array_t<unsigned char> img2) {
        cv::Mat mat1 = numpy_to_cv(img1);
        cv::Mat mat2 = numpy_to_cv(img2);
        return getPSNR(mat1, mat2);
    }, "Compute PSNR between two RGB images");

    // Referenčné metódy (OpenCV)
    m.def("rotate_nearest_ref", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_nearest_ref(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    m.def("rotate_bilinear_ref", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_bilinear_ref(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    m.def("rotate_bicubic_ref", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_bicubic_ref(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    m.def("rotate_lanczos_ref", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_lanczos_ref(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    // Manuálne metódy
    m.def("rotate_nearest_manual", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_nearest_manual(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    m.def("rotate_bilinear_manual", [](py::array_t<unsigned char> img, double angle, bool cut_corners) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_bilinear_manual(src, angle, cut_corners);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true);

    m.def("rotate_bicubic_manual", [](py::array_t<unsigned char> img, double angle, bool cut_corners, double sharpness) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_bicubic_manual(src, angle, cut_corners, sharpness);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true, py::arg("sharpness") = -0.75);

    m.def("rotate_lanczos_manual", [](py::array_t<unsigned char> img, double angle, bool cut_corners, int a) {
        cv::Mat src = numpy_to_cv(img);
        cv::Mat dst = rotate_lanczos_manual(src, angle, cut_corners, a);
        return cv_to_numpy(dst);
    }, py::arg("img"), py::arg("angle"), py::arg("cut_corners") = true, py::arg("a") = 4);


    // maximal inner rectangle: returns (width, height) as ints rounded
    m.def("get_max_inner_rect", [](int w, int h, double angle_deg) {
        double out_w = 0.0, out_h = 0.0;
        get_max_inner_rect((double)w, (double)h, angle_deg, out_w, out_h);
        int iw = std::max(0, (int)std::lround(out_w));
        int ih = std::max(0, (int)std::lround(out_h));
        return py::make_tuple(iw, ih);
    }, py::arg("width"), py::arg("height"), py::arg("angle_deg"));
}