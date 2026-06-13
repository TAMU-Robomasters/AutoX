#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/core.hpp>

#include <cstring>

#include "detector.h"
#include "subsystems/armor.h"
#include "subsystems/types.h"
#include "subsystems/icon_detection.h"
#include "subsystems/pnp.h"
#include "subsystems/video_source.h"

namespace py = pybind11;

// pybind11 has no built-in casters for OpenCV types, so expose Panel fields as
// numpy arrays via property accessors. All getters return owned copies so the
// Panel can be destroyed without dangling Python references.
static py::array_t<float> vec3_to_numpy(const cv::Vec3f& v) {
    py::array_t<float> arr(3);
    auto* d = arr.mutable_data();
    d[0] = v[0]; d[1] = v[1]; d[2] = v[2];
    return arr;
}

static cv::Vec3f numpy_to_vec3(
    py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    if (a.size() != 3) throw std::runtime_error("expected a 3-element array");
    auto d = a.unchecked<1>();
    return cv::Vec3f(d(0), d(1), d(2));
}

static py::array_t<float> point2f_to_numpy(const cv::Point2f& p) {
    py::array_t<float> arr(2);
    auto* d = arr.mutable_data();
    d[0] = p.x; d[1] = p.y;
    return arr;
}

static cv::Point2f numpy_to_point2f(
    py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    if (a.size() != 2) throw std::runtime_error("expected a 2-element array");
    auto d = a.unchecked<1>();
    return cv::Point2f(d(0), d(1));
}

static py::array_t<float> corners_to_numpy(const std::vector<cv::Point2f>& corners) {
    py::array_t<float> arr({(py::ssize_t)corners.size(), (py::ssize_t)2});
    auto d = arr.mutable_unchecked<2>();
    for (size_t i = 0; i < corners.size(); ++i) {
        d(i, 0) = corners[i].x;
        d(i, 1) = corners[i].y;
    }
    return arr;
}

static std::vector<cv::Point2f> numpy_to_corners(
    py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    auto buf = a.unchecked<2>();
    if (buf.shape(1) != 2) throw std::runtime_error("corners must have shape (N, 2)");
    std::vector<cv::Point2f> out(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out[i] = cv::Point2f(buf(i, 0), buf(i, 1));
    }
    return out;
}

// Adapts OpenCV-Python contours (list of (N, 1, 2) int32 numpy arrays) into
// std::vector<std::vector<cv::Point>> for the C++ pipeline.
static std::vector<Panel> contours_to_panels_py(
    const std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>& py_contours) {

    std::vector<std::vector<cv::Point>> contours;
    contours.reserve(py_contours.size());

    for (const auto& arr : py_contours) {
        auto buf = arr.unchecked<3>();  // shape (N, 1, 2)
        std::vector<cv::Point> c;
        c.reserve(buf.shape(0));
        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
            c.emplace_back(buf(i, 0, 0), buf(i, 0, 1));
        }
        contours.push_back(std::move(c));
    }

    return contours_to_panels(contours);
}

PYBIND11_MODULE(armor_panel_cpp, m) {
    m.doc() = "C++ detection pipeline for Armor Panel Classical";

    load_icons();
    set_icon_tolerance(50.0f);  // matches Python's config.icon_tolerance

    py::class_<Panel>(m, "Panel")
        .def(py::init<>())
        .def_readwrite("id", &Panel::id)
        .def_readwrite("area", &Panel::area)
        .def_readwrite("yaw", &Panel::yaw)
        .def_property("tvec",
            [](const Panel& p) { return vec3_to_numpy(p.tvec); },
            [](Panel& p, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                p.tvec = numpy_to_vec3(a);
            })
        .def_property("rvec",
            [](const Panel& p) { return vec3_to_numpy(p.rvec); },
            [](Panel& p, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                p.rvec = numpy_to_vec3(a);
            })
        .def_property("center",
            [](const Panel& p) { return point2f_to_numpy(p.center); },
            [](Panel& p, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                p.center = numpy_to_point2f(a);
            })
        .def_property("corners",
            [](const Panel& p) { return corners_to_numpy(p.corners); },
            [](Panel& p, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                p.corners = numpy_to_corners(a);
            });

    py::class_<Light>(m, "Light")
        .def(py::init<>())
        .def(py::init([](float cx, float cy, float w, float h, float angle) {
            Light l;
            l.cx = cx;
            l.cy = cy;
            l.w = w;
            l.h = h;
            l.angle = angle;
            return l;
        }), py::arg("cx"), py::arg("cy"), py::arg("w"), py::arg("h"), py::arg("angle"))
        .def_readwrite("cx", &Light::cx)
        .def_readwrite("cy", &Light::cy)
        .def_readwrite("w",  &Light::w)
        .def_readwrite("h",  &Light::h)
        .def_readwrite("angle", &Light::angle);

    m.def("detect_panels", &detect_panels,
          "Run the detection pipeline and return a list of Panels.");

    m.def("pair_lights", &pair_lights,
          "Pair lights into armor candidates.");

    m.def("contours_to_panels", &contours_to_panels_py,
          "Convert OpenCV-Python contours (list of (N,1,2) int32 arrays) to Panel objects.");

    m.def("solve_pnp", &solve_pnp,
          "Run PnP on a list of Panels to compute their pose.");

    m.def("video_source_init", &video_source_init,
          py::arg("device_id") = 0, py::arg("exposure"),
          "Open the USB camera and start the bufferless reader thread.");

    m.def("video_source_close", &video_source_close,
          "Stop the reader thread and release the camera.");

    m.def("set_reuse_stale_frame", &set_reuse_stale_frame, py::arg("enabled"),
          "Benchmark hack: when true, detect_panels() reuses the last frame instead of waiting for a new one.");

    m.def("get_last_frame",
          []() -> py::object {
              cv::Mat m = last_frame();
              if (m.empty()) return py::none();
              cv::Mat cont = m.isContinuous() ? m : m.clone();
              py::array_t<uint8_t> arr({cont.rows, cont.cols, cont.channels()});
              std::memcpy(arr.mutable_data(), cont.data, cont.total() * cont.elemSize());
              return arr;
          },
          "Return a numpy copy of the frame most recently consumed by detect_panels (for debug viz).");

    m.def("set_icon_tolerance", &set_icon_tolerance, py::arg("tol"),
          "Mean-XOR threshold for accepting an icon match. Higher = more permissive.");

    m.def("set_intrinsics",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> cam_matrix,
             py::array_t<double, py::array::c_style | py::array::forcecast> dist) {
              py::buffer_info ci = cam_matrix.request();
              py::buffer_info di = dist.request();
              if (ci.ndim != 2 || ci.shape[0] != 3 || ci.shape[1] != 3) {
                  throw std::runtime_error("cam_matrix must be a 3x3 array");
              }
              if (di.ndim != 1) {
                  throw std::runtime_error("dist must be a 1-D array");
              }
              cv::Mat cam(3, 3, CV_64F, ci.ptr);
              cv::Mat dst(1, (int)di.shape[0], CV_64F, di.ptr);
              set_intrinsics(cam, dst);
          },
          py::arg("cam_matrix"), py::arg("dist"),
          "Set camera intrinsics (cam_matrix 3x3, dist 1-D) for solve_pnp.");
}
