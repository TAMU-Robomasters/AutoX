#include "pnp.h"
#include "types.h"

#include <opencv2/calib3d.hpp>

static cv::Mat camera_matrix;
static cv::Mat dist_coeffs;

void set_intrinsics(const cv::Mat& cam, const cv::Mat& dist) {
    camera_matrix = cam.clone();
    dist_coeffs = dist.clone();
}

void solve_pnp(std::vector<Panel>& panels) {
    // Panel coordinates in cm; matches Python's PnP.py panel_coordinates.
    // +y is "up" in the model frame (matches Python's convention).
    static const std::vector<cv::Point3f> panel_coordinates = {
        cv::Point3f(-12.2f / 2.0f,  12.5f / 2.0f, 0.0f),
        cv::Point3f( 12.2f / 2.0f,  12.5f / 2.0f, 0.0f),
        cv::Point3f( 12.2f / 2.0f, -12.5f / 2.0f, 0.0f),
        cv::Point3f(-12.2f / 2.0f, -12.5f / 2.0f, 0.0f),
    };

    if (camera_matrix.empty() || dist_coeffs.empty()) {
        throw std::runtime_error("solve_pnp: intrinsics not set. Call set_intrinsics() first.");
    }

    for (auto& panel : panels) {
        if (panel.id == -1) continue;
        if (panel.corners.size() != 4) continue;

        cv::Vec3d rvec, tvec;
        bool success = cv::solvePnP(panel_coordinates,
                                    panel.corners,
                                    camera_matrix,
                                    dist_coeffs,
                                    rvec,
                                    tvec,
                                    false,
                                    cv::SOLVEPNP_ITERATIVE);
        if (success) {
            panel.rvec = cv::Vec3f((float)rvec[0], (float)rvec[1], (float)rvec[2]);
            // Camera -> ballistic frame swap to match Python's [tvec[0], tvec[2], -tvec[1]]
            panel.tvec = cv::Vec3f((float)tvec[0], (float)tvec[2], -(float)tvec[1]);
        }
    }
}
