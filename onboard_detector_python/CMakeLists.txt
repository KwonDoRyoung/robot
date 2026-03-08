cmake_minimum_required(VERSION 3.0.2)
project(onboard_detector_python)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  message_filters
  vision_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  GetDynamicObstacles.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    visualization_msgs
    cv_bridge
    message_filters
    vision_msgs
    message_runtime
)

catkin_install_python(
  PROGRAMS
    src/onboard_detector_python/detector_node.py
    src/onboard_detector_python/fake_detector_node.py
    scripts/yolo_detector/yolov11_detector_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch cfg rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)