cmake_minimum_required(VERSION 3.0.2)
project(avoidance_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/avoidance_policy_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
