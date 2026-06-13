#include "detector.h"

#include "subsystems/types.h"
#include "subsystems/armor.h"
#include "subsystems/frame_process.h"
#include "subsystems/icon_detection.h"
#include "subsystems/pnp.h"
#include "subsystems/video_source.h"

std::vector<Panel> detect_panels(char enemy_color) {
    cv::Mat frame = get_frame();
    if (frame.empty()) return {};

    std::vector<std::vector<cv::Point>> contours = process_frame(frame, enemy_color);

    std::vector<Light> lights = detect_lights(contours);
    std::vector<std::pair<Light, Light>> light_pairs = pair_lights(lights);
    std::vector<Panel> panels = define_panels(light_pairs);

    icon_detection(frame, panels);
    solve_pnp(panels);

    return panels;
}

std::vector<Panel> contours_to_panels(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<Light> lights = detect_lights(contours);
    std::vector<std::pair<Light, Light>> light_pairs = pair_lights(lights);
    return define_panels(light_pairs);
}
