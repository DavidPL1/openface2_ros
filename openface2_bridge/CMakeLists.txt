cmake_minimum_required(VERSION 2.8.3)
project(openface2_bridge)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  image_transport
  cv_bridge
  std_msgs
  geometry_msgs
  sensor_msgs
  roscpp
  tf2
  tf2_ros
  openface2_ros_msgs
)

find_package(Boost 1.5.9 REQUIRED COMPONENTS filesystem system)
find_package(OpenCV 3 REQUIRED)

find_library(OPENFACE_LD_LIB LandmarkDetector)
find_library(OPENFACE_FA_LIB FaceAnalyser)
find_library(OPENFACE_GA_LIB GazeAnalyser)
find_library(OPENFACE_UT_LIB Utilities)

set(OPENFACE_LIBS ${OPENFACE_LD_LIB} ${OPENFACE_FA_LIB} ${OPENFACE_GA_LIB} ${OPENFACE_UT_LIB})

find_path(OPENFACE_INCLUDE_DIR LandmarkDetectorModel.h PATH_SUFFIXES OpenFace)

add_definitions(-std=c++11)
add_definitions(-g -pg)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENFACE_INCLUDE_DIR}
  ${OpenBLAS_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}/boost
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(openface2_bridge
    src/openface2_ros.cpp
)

add_dependencies(openface2_bridge ${catkin_EXPORTED_TARGETS})

target_link_libraries(openface2_bridge
  ${catkin_LIBRARIES}
  ${OPENFACE_LIBS}
  openblas
  dlib
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(TARGETS openface2_bridge
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
