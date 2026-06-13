#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& arr) {
    // 1. Request buffer info to get dimensions, strides, and data pointer
    py::buffer_info info = arr.request();
    
    int rows = 0;
    int cols = 0;
    int type = CV_8UC1; // Default to single-channel 8-bit

    // 2. Determine matrix dimensions and type based on NumPy dimensions
    if (info.ndim == 2) {
        rows = info.shape[0];
        cols = info.shape[1];
        type = CV_8UC1;
    } else if (info.ndim == 3) {
        rows = info.shape[0];
        cols = info.shape[1];
        int channels = info.shape[2];
        
        // Ensure channels are supported by OpenCV
        if (channels > 4) {
            throw std::runtime_error("Only 1 to 4 channels are supported for cv::Mat");
        }
        type = CV_MAKETYPE(CV_8U, channels);
    } else {
        throw std::runtime_error("Unsupported number of dimensions. Must be 2 or 3.");
    }

    // 3. Construct cv::Mat without copying data
    // The py::capsule ensures the Python object's reference count is managed, 
    // keeping the memory alive for OpenCV's use.
    return cv::Mat(rows, cols, type, info.ptr, info.strides[0]);
}

std::vector<std::vector<cv::Point>> process_frame(cv::Mat frame, char color) {    
    cv::Mat enemy_color_mask;
    if (color == 'r') {
        cv::extractChannel(frame, enemy_color_mask, 2); // Extract the red channel
    } else {
        cv::extractChannel(frame, enemy_color_mask, 0); // Extract the blue channel
    }

    cv::threshold(enemy_color_mask, enemy_color_mask, 200, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(enemy_color_mask, enemy_color_mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(enemy_color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return contours;
    
}