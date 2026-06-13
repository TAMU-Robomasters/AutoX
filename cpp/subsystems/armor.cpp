#include "types.h"
#include "armor.h"
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

float armor_width_ratio = 5.5/13;  // match Python armor.py (armor_width_ration)

// Light-pairing tunables. Defaults mirror the historical hardcoded values so
// the module still works before Python pushes config in; CppDetectorModule
// overrides these from info.yaml's `classical:` block via set_pairing_params().
// See src/subsystems/vision/classical_detector/armor.py (pairing) for the
// authoritative meaning of each field.
struct PairingParams {
    float angle_diff_multiplier = 1.0f;
    float misalignment_multiplier = 2.0f;
    float expected_distance_multiplier = 0.5f;
    float height_ratio_multiplier = 1.0f;
    float angle_diff_thresh = 10.0f;          // max |angle1 - angle2| to pair
    float misalignment_thresh = 30.0f;        // max angle-with-horizon (deg)
    float height_ratio_thresh_lo = 0.5f;      // valid height-ratio band
    float height_ratio_thresh_hi = 2.0f;
    float score_thresh = 200.0f;              // reject pairs scoring >= this
};

static PairingParams g_pairing;

void set_pairing_params(float angle_diff_multiplier,
                        float misalignment_multiplier,
                        float expected_distance_multiplier,
                        float height_ratio_multiplier,
                        float angle_diff_thresh,
                        float misalignment_thresh,
                        float height_ratio_thresh_lo,
                        float height_ratio_thresh_hi,
                        float score_thresh) {
    g_pairing.angle_diff_multiplier = angle_diff_multiplier;
    g_pairing.misalignment_multiplier = misalignment_multiplier;
    g_pairing.expected_distance_multiplier = expected_distance_multiplier;
    g_pairing.height_ratio_multiplier = height_ratio_multiplier;
    g_pairing.angle_diff_thresh = angle_diff_thresh;
    g_pairing.misalignment_thresh = misalignment_thresh;
    g_pairing.height_ratio_thresh_lo = height_ratio_thresh_lo;
    g_pairing.height_ratio_thresh_hi = height_ratio_thresh_hi;
    g_pairing.score_thresh = score_thresh;
}

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
            if (angle_diff > g_pairing.angle_diff_thresh) {
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }
            float cx_diff = std::abs(light1.cx - light2.cx);
            float cy_diff = std::abs(light1.cy - light2.cy);
            float distance = std::sqrt(cx_diff * cx_diff + cy_diff * cy_diff);

            float angle_with_horizon = std::abs(std::atan2(cy_diff, cx_diff) * 180.0f / CV_PI);
            if (angle_with_horizon > g_pairing.misalignment_thresh) {
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }
            float height_ratio = light1.h / light2.h;
            float average_height = (light1.h + light2.h) / 2.0f;
            if (height_ratio < g_pairing.height_ratio_thresh_lo || height_ratio > g_pairing.height_ratio_thresh_hi){
                score_matrix[i][j] = placeholder; // Not a valid pair
                continue;
            }

            float expected_distance_comparison = std::abs((average_height / armor_width_ratio) - distance); // smaller better
            score_matrix[i][j] = angle_diff*g_pairing.angle_diff_multiplier + angle_with_horizon*g_pairing.misalignment_multiplier + expected_distance_comparison*g_pairing.expected_distance_multiplier + height_ratio*g_pairing.height_ratio_multiplier;
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
        if (best_i < 0 || min_score >= g_pairing.score_thresh) break;
        pairs.emplace_back(lights[best_i], lights[best_j]);
        used_panel.push_back(best_i);
        used_panel.push_back(best_j);
    }

    
    return pairs;
}

std::vector<Panel> define_panels (std::vector<std::pair<Light, Light>> light_pairs){
    std::vector<Panel> panels;
    panels.reserve(light_pairs.size());
    const float armor_height_ratio = 12.5f / 5.2f;  // match Python armor.py


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