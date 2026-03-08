#!/usr/bin/env python3
"""
fake_detector_node.py
---------------------
ROS node entry point for FakeDetector (Python port of fakeDetector).
"""
import rospy
from onboard_detector_python.fake_detector import FakeDetector


def main():
    rospy.init_node("fake_detector_python", anonymous=False)
    detector = FakeDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
