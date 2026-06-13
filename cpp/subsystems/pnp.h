#ifndef PNP_H
#define PNP_H

#include <vector>

#include <opencv2/core.hpp>

#include "types.h"

void set_intrinsics(const cv::Mat& cam_matrix, const cv::Mat& dist_coeffs);
void solve_pnp(std::vector<Panel>& panels);

#endif
