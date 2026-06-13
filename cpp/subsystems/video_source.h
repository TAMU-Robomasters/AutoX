#ifndef VIDEO_SOURCE_H
#define VIDEO_SOURCE_H

#include <utility>

#include <opencv2/core.hpp>

// Open the USB camera at device_id, configure for 1280x720 MJPG @ 30fps with
// the given exposure, and start a background reader thread that keeps only the
// most recent frame. Mirrors Python's BufferlesCvCapture in subsystems/video_source.py.
void video_source_init(int device_id, double exposure);

// Stop the reader thread and release the camera. Safe to call multiple times.
void video_source_close();

// Pop the next frame from the bufferless queue. Blocks if none is available yet.
// Also stashes the result so last_frame() can hand it to a debug viewer.
cv::Mat get_frame();

// Non-blocking newest-frame accessor (mirrors Python's FrameReader.latest()):
// returns {copy-of-newest-frame, monotonic-seq} without blocking or consuming.
// Empty Mat (seq 0) until the first frame arrives. The caller dedups on seq to
// decide whether a new frame is available. Also stashes the frame for last_frame().
std::pair<cv::Mat, uint64_t> read_latest_frame();

// Benchmark hack: when true, get_frame() does not block waiting for a fresh frame.
// If no new frame is available it returns the last one again. Defaults to false.
void set_reuse_stale_frame(bool enabled);

// Return a refcounted handle to the frame most recently returned by get_frame().
// Empty Mat if get_frame has not been called.
cv::Mat last_frame();

#endif
