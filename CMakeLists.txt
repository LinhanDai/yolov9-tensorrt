cmake_minimum_required(VERSION 3.10)
project(yolov9)

option(RELEASE          "build Yolov9 lib release"   OFF)

if(RELEASE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -DNODEBUG -O3 -Wall")
else()
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g") ### open compiler debug flag
endif()

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#opencv
find_package(OpenCV REQUIRED)

# cuda
find_package(CUDA)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
link_directories(/usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
include_directories(/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)


#yolov9 source
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB LIB_SOURCES  src/*.cu
        src/*.cpp
        include/*.h)

#add cuda lib
CUDA_ADD_LIBRARY(yolov9 SHARED ${LIB_SOURCES})

target_link_libraries(yolov9
        nvinfer
        nvonnxparser
        pthread
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES})

message("OpenCV: ${OpenCV_LIBRARIES}")
target_include_directories(yolov9 PRIVATE include/
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include)

add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(demo yolov9)
