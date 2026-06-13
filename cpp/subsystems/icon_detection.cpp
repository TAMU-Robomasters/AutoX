#include <opencv4/opencv2/opencv.hpp>
#include "icon_detection.h"

#include "types.h"
#include "armor.h"

static std::vector<cv::Mat> icon_stack;
static float icon_tolerance = 50.0f;

void load_icons() {
    icon_stack.clear();
    // Match Python: load icons as plain grayscale (no thresholding).
    icon_stack.push_back(cv::imread("/home/orin/repos/Armor-Panel-Classical/icons/complete_sentry_icon.jpeg", cv::IMREAD_GRAYSCALE));
    icon_stack.push_back(cv::imread("/home/orin/repos/Armor-Panel-Classical/icons/cropped_sentry.png", cv::IMREAD_GRAYSCALE));
    icon_stack.push_back(cv::imread("/home/orin/repos/Armor-Panel-Classical/icons/cropped_1.png", cv::IMREAD_GRAYSCALE));
    icon_stack.push_back(cv::imread("/home/orin/repos/Armor-Panel-Classical/icons/cropped_3.png", cv::IMREAD_GRAYSCALE));

    // Each icon needs to be 300x300 to XOR against the cropped panel.
    for (auto& icon : icon_stack) {
        if (icon.empty()) continue;
        if (icon.cols != 300 || icon.rows != 300) {
            cv::Mat resized;
            cv::resize(icon, resized, cv::Size(300, 300));
            icon = resized;
        }
    }
}

void set_icon_tolerance(float tol) {
    icon_tolerance = tol;
}

void icon_detection(const cv::Mat& input_image, std::vector<Panel>& panels) {
    cv::Mat gray_image;
    cv::Mat warped_image;
    cv::Mat xor_img;
    cv::Mat cropped_icon;
    cv::Mat cropped_flipped;
    std::vector<std::vector<cv::Point>> contours;

    static const std::vector<cv::Point2f> dst_corners = {
        cv::Point2f(0,   0),
        cv::Point2f(300, 0),
        cv::Point2f(300, 300),
        cv::Point2f(0,   300),
    };

    for (auto& panel : panels) {
        panel.id = -1;
        if (panel.corners.size() != 4) continue;

        cv::Mat M = cv::getPerspectiveTransform(panel.corners, dst_corners);
        cv::warpPerspective(input_image, warped_image, M, cv::Size(300, 300));
        cv::cvtColor(warped_image, gray_image, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(gray_image, gray_image, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -5);

        // Match Python: RETR_TREE, then skip the first 2 contours and take the
        // largest of the remainder (Python: `max(icon_contours[2:], key=cv.contourArea)`).
        cv::findContours(gray_image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        if (contours.size() < 3) continue;

        const std::vector<cv::Point>* largest = nullptr;
        double max_area = -1.0;
        for (size_t i = 2; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                largest = &contours[i];
            }
        }
        if (!largest) continue;

        cv::Rect bbox = cv::boundingRect(*largest);
        if (bbox.width <= 0 || bbox.height <= 0) continue;

        cv::resize(gray_image(bbox), cropped_icon, cv::Size(300, 300));
        // Python: cv.flip(cropped, 0)  -- vertical flip
        cv::flip(cropped_icon, cropped_flipped, 0);

        double best_score = std::numeric_limits<double>::infinity();
        int best_id = -1;
        for (size_t i = 0; i < icon_stack.size(); ++i) {
            if (icon_stack[i].empty()) continue;
            cv::bitwise_xor(cropped_flipped, icon_stack[i], xor_img);
            double score = cv::mean(xor_img)[0];
            if (score < best_score) {
                best_score = score;
                best_id = (int)i;
            }
        }

        if (best_score <= icon_tolerance) {
            panel.id = best_id;
        }
    }
}
