#include "video_source.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <opencv2/videoio.hpp>

namespace {

class BufferlessCapture {
public:
    BufferlessCapture(int device_id, double exposure) {
        cap_.open(device_id, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Could not open video.");
        }

        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap_.set(cv::CAP_PROP_EXPOSURE, exposure);
        cap_.set(cv::CAP_PROP_FPS, 90);
        thread_ = std::thread(&BufferlessCapture::reader_loop, this);
    }

    ~BufferlessCapture() {
        stop_.store(true);
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        cap_.release();
    }

    cv::Mat read() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return has_frame_ || stop_.load(); });
        if (!has_frame_) return {};
        cv::Mat out = std::move(latest_);
        has_frame_ = false;
        return out;
    }

    cv::Mat try_read() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!has_frame_) return {};
        cv::Mat out = std::move(latest_);
        has_frame_ = false;
        return out;
    }

private:
    void reader_loop() {
        while (!stop_.load()) {
            cv::Mat frame;
            if (!cap_.read(frame)) {
                stop_.store(true);
                cv_.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                latest_ = std::move(frame);
                has_frame_ = true;
            }
            cv_.notify_one();
        }
    }

    cv::VideoCapture cap_;
    std::mutex mu_;
    std::condition_variable cv_;
    cv::Mat latest_;
    bool has_frame_ = false;
    std::thread thread_;
    std::atomic<bool> stop_{false};
};

std::unique_ptr<BufferlessCapture> capture;
cv::Mat last_frame_storage;
bool reuse_stale = false;

}  // namespace

void video_source_init(int device_id, double exposure) {
    capture = std::make_unique<BufferlessCapture>(device_id, exposure);
}

void video_source_close() {
    capture.reset();
}

cv::Mat get_frame() {
    if (!capture) {
        throw std::runtime_error("video_source_init has not been called");
    }
    if (reuse_stale) {
        cv::Mat fresh = capture->try_read();
        if (!fresh.empty()) {
            last_frame_storage = fresh;
        } else if (last_frame_storage.empty()) {
            // No stale frame yet — block once so the pipeline gets a real first frame.
            last_frame_storage = capture->read();
        }
        // else: reuse the previous last_frame_storage
    } else {
        last_frame_storage = capture->read();
    }
    return last_frame_storage;
}

void set_reuse_stale_frame(bool enabled) {
    reuse_stale = enabled;
}

cv::Mat last_frame() {
    return last_frame_storage;
}
