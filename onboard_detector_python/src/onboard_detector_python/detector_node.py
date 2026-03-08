#!/usr/bin/env python3
"""
detector_node.py
----------------
ROS node entry point for DynamicDetector (Python port of onboard_detector).
"""
import rospy
from onboard_detector_python.dynamic_detector import DynamicDetector


def main():
    rospy.init_node("onboard_detector_python", anonymous=False)
    detector = DynamicDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
