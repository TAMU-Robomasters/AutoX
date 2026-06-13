#ifndef ARMOR_PANEL_TYPES_H
#define ARMOR_PANEL_TYPES_H

#include <opencv2/core.hpp>

struct Panel {
        std::vector<cv::Point2f> corners;
        cv::Point2f center;
        cv::Vec3f tvec;
        cv::Vec3f rvec;
        int id = 0;
        float area = 0.0f;
        float yaw = 0.0f;
};

struct Light {
    cv::RotatedRect bbox;
    float cx , cy, w, h, angle;
};

#endif
