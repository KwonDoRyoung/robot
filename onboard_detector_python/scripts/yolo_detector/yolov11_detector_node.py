#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
import numpy as np
from yolov11_detector import *


def main():
	rospy.init_node("yolov11_detector_node")
	yolo_detector()
	rospy.spin()

if __name__=="__main__":
	main()
	
