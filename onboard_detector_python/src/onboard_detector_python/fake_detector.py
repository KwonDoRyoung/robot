#!/usr/bin/env python3
"""
fake_detector.py
----------------
Python port of onboardDetector::fakeDetector.
Uses gazebo_msgs/ModelStates to provide ground-truth dynamic obstacle info.
"""

import math
from collections import deque
from copy import deepcopy

import numpy as np
import rospy

from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header

from onboard_detector_python.utils import Box3D, rpy_from_quaternion, angle_between_vectors


PI = math.pi


class FakeDetector:
    """Python port of onboardDetector::fakeDetector."""

    HINT = "[Fake Detector]"

    def __init__(self):
        # Parameters
        self.target_obstacles = rospy.get_param("target_obstacle", ["person", "obstacle"])
        self.color_distance = rospy.get_param("color_distance", 5.0)
        odom_topic = rospy.get_param("odom_topic", "/CERLAB/quadcopter/odom")
        self.hist_size = int(rospy.get_param("history_size", 5))

        rospy.loginfo(f"{self.HINT}: target_obstacle = {self.target_obstacles}")
        rospy.loginfo(f"{self.HINT}: color_distance = {self.color_distance}")
        rospy.loginfo(f"{self.HINT}: odom_topic = {odom_topic}")
        rospy.loginfo(f"{self.HINT}: history_size = {self.hist_size}")

        # State
        self.first_time = True
        self.target_indices = []
        self.obstacle_msg = []        # List[Box3D] — current frame
        self.last_ob_vec = []         # List[Box3D] — previous frame (for velocity)
        self.last_time_vec = []       # List[rospy.Time]
        self.last_time_vel = []       # List[List[float]]  [vx, vy, vz]
        self.obstacle_hist = []       # List[deque[Box3D]]
        self.odom = Odometry()

        # Publishers
        self._pub_hist_traj = rospy.Publisher(
            "onboard_detector/history_trajectories", MarkerArray, queue_size=10)
        self._pub_vis = rospy.Publisher(
            "onboard_detector/GT_obstacle_bbox", MarkerArray, queue_size=10)
        self._vis_msg = MarkerArray()

        # Subscribers
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._state_cb, queue_size=10)
        rospy.Subscriber(odom_topic, Odometry,
                         self._odom_cb, queue_size=10)

        # Timers
        rospy.Timer(rospy.Duration(0.033), self._hist_cb)
        rospy.Timer(rospy.Duration(0.05), self._vis_cb)

        # Service
        from onboard_detector_python.srv import GetDynamicObstacles
        rospy.Service("fake_detector/getDynamicObstacles",
                      GetDynamicObstacles, self._get_dynamic_obstacles_srv)

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def _get_dynamic_obstacles_srv(self, req):
        from onboard_detector_python.srv import GetDynamicObstaclesResponse
        from geometry_msgs.msg import Vector3

        res = GetDynamicObstaclesResponse()
        curr_pos = np.array([req.current_position.x, req.current_position.y, req.current_position.z])

        obstacles = []
        for bbox in self.obstacle_msg:
            obs_pos = np.array([bbox.x, bbox.y, bbox.z])
            diff = curr_pos - obs_pos
            diff[2] = 0.0
            dist = np.linalg.norm(diff)
            if dist <= req.range:
                obstacles.append((dist, bbox))

        obstacles.sort(key=lambda x: x[0])
        for dist, bbox in obstacles:
            pos = Vector3(x=bbox.x, y=bbox.y, z=bbox.z)
            vel = Vector3(x=bbox.Vx, y=bbox.Vy, z=bbox.Vz)
            size = Vector3(x=bbox.x_width, y=bbox.y_width, z=bbox.z_width)
            res.position.append(pos)
            res.velocity.append(vel)
            res.size.append(size)
        return res

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _state_cb(self, all_states):
        """Process Gazebo model states: extract target obstacle positions & velocities."""
        update = False

        if self.first_time:
            self.target_indices = self._find_target_index(all_states.name)
            self.first_time = False

        ob_vec = []
        for i, idx in enumerate(self.target_indices):
            name = all_states.name[idx]
            pose = all_states.pose[idx]
            ob = Box3D()
            ob.x = pose.position.x
            ob.y = pose.position.y
            # person: add 0.9m height offset to center
            if len(name) >= 6 and name[:6] == "person":
                ob.z = pose.position.z + 0.9
            else:
                ob.z = pose.position.z

            # Velocity estimation
            if not self.last_ob_vec:
                ob.Vx = 0.0; ob.Vy = 0.0; ob.Vz = 0.0
                self.last_time_vec.append(rospy.Time.now())
                self.last_time_vel.append([0.0, 0.0, 0.0])
                update = True
            else:
                curr_time = rospy.Time.now()
                dT = (curr_time - self.last_time_vec[i]).to_sec()
                if dT >= 0.1:
                    vx = (ob.x - self.last_ob_vec[i].x) / dT
                    vy = (ob.y - self.last_ob_vec[i].y) / dT
                    vz = (ob.z - self.last_ob_vec[i].z) / dT
                    ob.Vx = vx; ob.Vy = vy; ob.Vz = vz
                    self.last_time_vel[i] = [vx, vy, vz]
                    self.last_time_vec[i] = rospy.Time.now()
                    update = True
                else:
                    ob.Vx = self.last_time_vel[i][0]
                    ob.Vy = self.last_time_vel[i][1]
                    ob.Vz = self.last_time_vel[i][2]

            # Parse size from Gazebo model name (encoded as last 9 chars: XXXyyyzzz)
            try:
                x_size = float(name[-1-1-3*3: -1-3*2])
                y_size = float(name[-1-3*2: -1-3*1])
                z_size = float(name[-3:])
            except (ValueError, IndexError):
                x_size = y_size = z_size = 0.5  # fallback

            ob.x_width = x_size
            ob.y_width = y_size
            ob.z_width = z_size
            ob_vec.append(ob)

        if update:
            self.last_ob_vec = ob_vec
        self.obstacle_msg = ob_vec

    def _odom_cb(self, odom_msg):
        self.odom = odom_msg

    def _hist_cb(self, event):
        """Maintain sliding window history of obstacle states."""
        if not self.obstacle_hist:
            self.obstacle_hist = [deque() for _ in range(len(self.obstacle_msg))]
        for i, ob in enumerate(self.obstacle_msg):
            if i >= len(self.obstacle_hist):
                self.obstacle_hist.append(deque())
            if len(self.obstacle_hist[i]) >= self.hist_size:
                self.obstacle_hist[i].pop()
            self.obstacle_hist[i].appendleft(deepcopy(ob))

    def _vis_cb(self, event):
        self._publish_history_traj()
        self._publish_visualization()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _make_pt(self, x, y, z):
        p = Point(); p.x = x; p.y = y; p.z = z
        return p

    def _update_vis_msg(self):
        bbox_vec = []
        for ob_idx, ob in enumerate(self.obstacle_msg):
            x, y, z = ob.x, ob.y, ob.z
            xw, yw, zw = ob.x_width/2, ob.y_width/2, ob.z_width/2

            p = [None] * 8
            p[0] = self._make_pt(x+xw, y+yw, z+zw)
            p[1] = self._make_pt(x-xw, y+yw, z+zw)
            p[2] = self._make_pt(x+xw, y-yw, z+zw)
            p[3] = self._make_pt(x-xw, y-yw, z+zw)
            p[4] = self._make_pt(x+xw, y+yw, z-zw)
            p[5] = self._make_pt(x-xw, y+yw, z-zw)
            p[6] = self._make_pt(x+xw, y-yw, z-zw)
            p[7] = self._make_pt(x-xw, y-yw, z-zw)

            lines = [
                (p[0], p[1]), (p[0], p[2]), (p[1], p[3]), (p[2], p[3]),
                (p[0], p[4]), (p[1], p[5]), (p[2], p[6]), (p[3], p[7]),
                (p[4], p[5]), (p[4], p[6]), (p[5], p[7]), (p[6], p[7]),
            ]

            in_range = self._is_obstacle_in_sensor_range(ob, PI)
            ns = f"GT osbtacles{ob_idx}"
            for count, (pa, pb) in enumerate(lines):
                m = Marker()
                m.header.frame_id = "map"
                m.ns = ns
                m.id = count
                m.type = Marker.LINE_LIST
                m.lifetime = rospy.Duration(0.5)
                m.scale.x = m.scale.y = m.scale.z = 0.05
                m.color.a = 1.0
                m.color.r = 1.0 if in_range else 0.0
                m.color.g = 0.0 if in_range else 1.0
                m.color.b = 0.0
                m.points = [pa, pb]
                bbox_vec.append(m)

        self._vis_msg.markers = bbox_vec

    def _publish_visualization(self):
        self._update_vis_msg()
        self._pub_vis.publish(self._vis_msg)

    def _publish_history_traj(self):
        if not self.obstacle_hist:
            return
        traj_msg = MarkerArray()
        count = 0
        for i, hist in enumerate(self.obstacle_hist):
            if not hist:
                continue
            if not self._is_obstacle_in_sensor_range(hist[0], 2 * PI):
                continue
            traj = Marker()
            traj.header.frame_id = "map"
            traj.header.stamp = rospy.Time.now()
            traj.ns = "fake_detector"
            traj.id = count
            traj.type = Marker.LINE_STRIP
            traj.scale.x = traj.scale.y = traj.scale.z = 0.1
            traj.color.a = 1.0; traj.color.r = 0.0; traj.color.g = 1.0; traj.color.b = 0.0
            for b in hist:
                traj.points.append(self._make_pt(b.x, b.y, b.z))
            traj_msg.markers.append(traj)
            count += 1
        self._pub_hist_traj.publish(traj_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_target_index(self, model_names):
        indices = []
        for i, name in enumerate(model_names):
            for target in self.target_obstacles:
                if name[:len(target)] == target:
                    indices.append(i)
                    break
        return indices

    def _is_obstacle_in_sensor_range(self, ob, fov):
        odom = self.odom
        robot_pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ])
        obs_pos = np.array([ob.x, ob.y, ob.z])
        diff = obs_pos - robot_pos
        diff[2] = 0.0
        distance = np.linalg.norm(diff)

        quat = odom.pose.pose.orientation
        yaw = rpy_from_quaternion(quat)
        direction = np.array([math.cos(yaw), math.sin(yaw), 0.0])

        angle = angle_between_vectors(direction, diff)
        return angle <= fov / 2.0 and distance <= self.color_distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obstacles(self, robot_size=None):
        """Return all obstacles expanded by robot_size."""
        if robot_size is None:
            robot_size = np.zeros(3)
        result = []
        for ob in self.obstacle_msg:
            b = deepcopy(ob)
            b.x_width += robot_size[0]
            b.y_width += robot_size[1]
            b.z_width += robot_size[2]
            result.append(b)
        return result

    def get_obstacles_in_sensor_range(self, fov, robot_size=None):
        """Return obstacles within fov angle and color_distance."""
        if robot_size is None:
            robot_size = np.zeros(3)
        result = []
        for ob in self.obstacle_msg:
            if self._is_obstacle_in_sensor_range(ob, fov):
                b = deepcopy(ob)
                b.x_width += robot_size[0]
                b.y_width += robot_size[1]
                b.z_width += robot_size[2]
                result.append(b)
        return result

    def get_dynamic_obstacles_hist(self, robot_size=None):
        """Return (pos_hist, vel_hist, size_hist) for obstacles in sensor range."""
        if robot_size is None:
            robot_size = np.zeros(3)
        pos_hist, vel_hist, size_hist = [], [], []
        for hist in self.obstacle_hist:
            if not hist:
                continue
            if not self._is_obstacle_in_sensor_range(hist[0], 2 * PI):
                continue
            op, ov, os_ = [], [], []
            for b in hist:
                op.append(np.array([b.x, b.y, b.z]))
                ov.append(np.array([b.Vx, b.Vy, 0.0]))
                os_.append(np.array([b.x_width + robot_size[0],
                                     b.y_width + robot_size[1],
                                     b.z_width + robot_size[2]]))
            pos_hist.append(op); vel_hist.append(ov); size_hist.append(os_)
        return pos_hist, vel_hist, size_hist
