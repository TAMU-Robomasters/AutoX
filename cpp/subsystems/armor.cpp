#include "types.h"
#include "armor.h"
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

float armor_width_ratio = 5.7/12.2;
float angle_diff_multiplier = 1;
float misalignment_multiplier = 2;
float expected_distance_multiplier = 0.5;
float height_ratio_multiplier = 1;

float placeholder = 10000000000;

static float deg_rad(float deg) {
    return deg * (float)CV_PI / 180.0f;
}

std::vector<Light> detect_lights(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<Light> lights;
    for (const auto& contour : contours) {
        if (contour.size() < 3) continue;  // minAreaRect needs >= 3 points

        cv::RotatedRect rect = cv::minAreaRect(contour);

        float rw = rect.size.width;
        float rh = rect.size.height;
        float h = std::max(rw, rh);
        float w = std::min(rw, rh);
        float angle = rect.angle;
        if (rh > rw) angle += 90.0f;
        if (std::abs(angle - 90.0f) > 45.0f) continue;

        Light light;
        light.cx = rect.center.x;
        light.cy = rect.center.y;
        light.w = w;
        light.h = h;
        light.angle = angle;
        lights.push_back(light);
    }
    return lights;
}

std::vector<std::pair<Light, Light>> pair_lights(const std::vector<Light>& lights) {
    std::vector<std::pair<Light, Light>> pairs;
    int num_lights = lights.size();
    float score_matrix[num_lights][num_lights];
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            const Light& light1 = lights[i];
            const Light& light2 = lights[j];

            // Check if the two lights can form a valid pair based on their properties
            float angle_diff = std::abs(light1.angle - light2.angle);
            if (angle_diff > 10.0f) {
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }
            float cx_diff = std::abs(light1.cx - light2.cx);
            float cy_diff = std::abs(light1.cy - light2.cy);
            float distance = std::sqrt(cx_diff * cx_diff + cy_diff * cy_diff);

            float angle_with_horizon = std::abs(std::atan2(cy_diff, cx_diff) * 180.0f / CV_PI);
            if (angle_with_horizon > 30.0f) {
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }
            float height_ratio = light1.h / light2.h;
            float average_height = (light1.h + light2.h) / 2.0f;
            if (height_ratio < 0.5f || height_ratio > 2.0){
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }
            
            float expected_distance_comparison = std::abs((average_height / armor_width_ratio) - distance); // smaller better
            score_matrix[i][j] = angle_diff* angle_diff_multiplier + angle_with_horizon*misalignment_multiplier + expected_distance_comparison*expected_distance_multiplier + height_ratio*height_ratio_multiplier;
        }
    }
    std::vector<int> used_panel;
    while (true) {
        float min_score = placeholder;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < num_lights; ++i) {
            if (std::find(used_panel.begin(), used_panel.end(), i) != used_panel.end()) continue;
            for (int j = i + 1; j < num_lights; ++j) {
                if (std::find(used_panel.begin(), used_panel.end(), j) != used_panel.end()) continue;
                if (score_matrix[i][j] < min_score) {
                    min_score = score_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_i < 0 || min_score >= 200.0f) break;
        pairs.emplace_back(lights[best_i], lights[best_j]);
        used_panel.push_back(best_i);
        used_panel.push_back(best_j);
    }

    
    return pairs;
}

std::vector<Panel> define_panels (std::vector<std::pair<Light, Light>> light_pairs){
    std::vector<Panel> panels;
    panels.reserve(light_pairs.size());
    const float armor_height_ratio = 12.5f / 5.7f;


    for (const auto& pair : light_pairs) {
        const Light* left;
        const Light* right;
        if (pair.first.cx <= pair.second.cx) {
            left  = &pair.first;
            right = &pair.second;
        } else {
            left  = &pair.second;
            right = &pair.first;
        }

        const float l_cos = std::cos(deg_rad(left->angle));
        const float l_sin = std::sin(deg_rad(left->angle));
        const float r_cos = std::cos(deg_rad(right->angle));
        const float r_sin = std::sin(deg_rad(right->angle));

        const float l_half_h = left->h  * armor_height_ratio * 0.5f;
        const float r_half_h = right->h * armor_height_ratio * 0.5f;

        cv::Point top_left(
            (int)(left->cx + left->w * 0.5f - l_half_h * l_cos),
            (int)(left->cy - l_half_h * l_sin)
        );
        cv::Point top_right(
            (int)(right->cx - right->w * 0.5f - r_half_h * r_cos),
            (int)(right->cy - r_half_h * r_sin)
        );
        cv::Point bottom_left(
            (int)(left->cx + left->w * 0.5f + l_half_h * l_cos),
            (int)(left->cy + l_half_h * l_sin)
        );
        cv::Point bottom_right(
            (int)(right->cx - right->w * 0.5f + r_half_h * r_cos),
            (int)(right->cy + r_half_h * r_sin)
        );

        std::vector<cv::Point> pts = { top_left, top_right, bottom_right, bottom_left };

        cv::Point2f panel_center(
            (top_left.x + top_right.x + bottom_right.x + bottom_left.x) / 4.0f,
            (top_left.y + top_right.y + bottom_right.y + bottom_left.y) / 4.0f
        );

        float area = (float)cv::contourArea(pts);

        Panel panel;
        panel.corners = {
            top_left,
            top_right,
            bottom_right,
            bottom_left
        };
        panel.center = panel_center;
        panel.area = area;
        panels.push_back(panel);
    }
    return panels;
}