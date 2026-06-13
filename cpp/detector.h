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

// Result of a non-blocking detection tick.
struct DetectResult {
    long seq = 0;          // sequence number of the newest available frame
    bool is_new = false;   // true iff seq advanced past last_seq (a frame was processed)
    std::vector<Panel> panels;  // empty unless is_new
};

// Non-blocking variant of detect_panels: grabs the newest frame WITHOUT blocking.
// If its seq has not advanced past last_seq, returns {seq, false, {}} so the
// caller can predict-only between camera frames. Otherwise runs the full pipeline.
DetectResult detect_panels_latest(char enemy_color, long last_seq);

#endif
