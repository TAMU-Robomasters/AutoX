cmake_minimum_required(VERSION 3.18)
project(armor_panel_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vpi 3 REQUIRED PATHS /opt/nvidia/vpi3/lib/aarch64-linux-gnu/cmake)
find_package(Threads REQUIRED)

pybind11_add_module(armor_panel_cpp
    bindings.cpp
    detector.cpp
    subsystems/video_source.cpp
    subsystems/frame_process.cpp
    subsystems/armor.cpp
    subsystems/icon_detection.cpp
    subsystems/pnp.cpp
)

target_include_directories(armor_panel_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(armor_panel_cpp PRIVATE ${OpenCV_LIBS} Threads::Threads)

install(TARGETS armor_panel_cpp
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
