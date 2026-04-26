cmake_minimum_required(VERSION 3.22.1)

project(scrfdncnn)

# ---------- OpenCV ----------
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android/sdk/native/jni)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# ---------- ncnn ----------
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# ---------- build ----------
add_library(scrfdncnn SHARED
        scrfdncnn.cpp
        scrfd.cpp
        recognizer.cpp
        ndkcamera.cpp
)

target_link_libraries(scrfdncnn
        ncnn
        ${OpenCV_LIBS}
        camera2ndk
        mediandk
        log
)