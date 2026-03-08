cmake_minimum_required(VERSION 3.0.2)
project(robot_slam)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  nodelet
)

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp rospy
    std_msgs geometry_msgs nav_msgs sensor_msgs
    tf2 tf2_ros tf2_geometry_msgs
    pcl_ros pcl_conversions
    nodelet
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Point cloud pre-filter node
add_executable(pointcloud_filter_node src/pointcloud_filter_node.cpp)
target_link_libraries(pointcloud_filter_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Install scripts
catkin_install_python(PROGRAMS
  scripts/pose_correction_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch and config files
install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
