#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>

#include <opencv2/core.hpp>

#include "subsystems/types.h"
#include "subsystems/frame_process.h"
#include "subsystems/armor.h"
#include "subsystems/icon_detection.h"
#include "subsystems/pnp.h"
#include "subsystems/video_source.h"

std::vector<Panel> detect_panels(char enemy_color);
std::vector<Panel> contours_to_panels(const std::vector<std::vector<cv::Point>>& contours);

#endif
