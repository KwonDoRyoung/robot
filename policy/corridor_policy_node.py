#!/usr/bin/env python3
"""
corridor_policy_node.py
-----------------------
복도 주행 정책 ROS1 노드

구독:
  /odom_3d                             (nav_msgs/Odometry)   로봇 위치/자세
  /onboard_detector/tracked_bboxes     (MarkerArray)         트래킹된 장애물
  /onboard_detector/dynamic_bboxes     (MarkerArray)         동적 장애물

발행:
  /cmd_vel                             (geometry_msgs/Twist) 속도 명령

파라미터 (ROS param):
  ~normal_speed        (float, 0.3)    정상 전진 속도 m/s
  ~slow_speed          (float, 0.1)    서행 속도 m/s
  ~lateral_speed       (float, 0.2)    횡이동 속도 m/s
  ~approach_vel_thresh (float, 0.15)   접근 판정 속도 임계값 m/s
  ~approach_dist_thresh(float, 3.0)    접근 판정 거리 임계값 m
  ~danger_zone_y       (list, [])      위험 구간 [(y_min, y_max), ...] world frame
  ~policy_rate         (float, 20.0)   정책 실행 주기 Hz
"""

import math
import threading
import numpy as np
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corridor_policy import CorridorPolicy, SPEED_NORMAL, SPEED_SLOW, LATERAL_SPEED
from corridor_policy import APPROACH_VEL_THRESH, APPROACH_DIST_THRESH


# Marker.type → 이름 매핑 (LINE_LIST = 5)
_BOX_MARKER_TYPE = 5


def _extract_box_from_marker(m):
    """MarkerArray의 LINE_LIST Marker에서 bbox 중심/속도 정보를 간이 추출."""
    class _Box:
        pass
    b = _Box()
    b.x = m.pose.position.x
    b.y = m.pose.position.y
    b.z = m.pose.position.z
    # velocity는 Marker에 직접 없으므로 0으로 초기화
    # (tracked_bboxes에서 text marker를 같이 파싱하는 방식 대신
    #  /onboard_detector/velocity_visualizaton topic 활용)
    b.Vx = 0.0
    b.Vy = 0.0
    return b


class CorridorPolicyNode:

    def __init__(self):
        rospy.init_node("corridor_policy_node")

        # 파라미터 로드
        self.policy = CorridorPolicy()
        self.policy.danger_zones = []  # 기본값 초기화

        import corridor_policy as cp
        cp.SPEED_NORMAL         = rospy.get_param("~normal_speed",         0.3)
        cp.SPEED_SLOW           = rospy.get_param("~slow_speed",           0.1)
        cp.LATERAL_SPEED        = rospy.get_param("~lateral_speed",        0.2)
        cp.APPROACH_VEL_THRESH  = rospy.get_param("~approach_vel_thresh",  0.15)
        cp.APPROACH_DIST_THRESH = rospy.get_param("~approach_dist_thresh", 3.0)

        dz = rospy.get_param("~danger_zone_y", [])
        if dz:
            # 파라미터 형식: [y_min1, y_max1, y_min2, y_max2, ...]
            zones = [(dz[i], dz[i+1]) for i in range(0, len(dz)-1, 2)]
            self.policy.set_danger_zones(zones)
            rospy.loginfo(f"[Policy] danger zones: {zones}")
        else:
            rospy.logwarn("[Policy] No danger_zone_y set. Scenario 1 inactive.")

        self._rate_hz = rospy.get_param("~policy_rate", 20.0)
        self._lock = threading.Lock()

        # 속도 벡터 저장 (vel_vis topic에서)
        self._vel_map = {}   # bbox 인덱스 → (Vx, Vy)

        # Publisher
        self._pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber("/odom_3d", Odometry,
                         self._odom_cb, queue_size=5)
        rospy.Subscriber("/onboard_detector/tracked_bboxes", MarkerArray,
                         self._tracked_cb, queue_size=5)
        rospy.Subscriber("/onboard_detector/dynamic_bboxes", MarkerArray,
                         self._dynamic_cb, queue_size=5)
        # 속도 시각화 마커 (TEXT_VIEW_FACING → Vx, Vy 포함)
        rospy.Subscriber("/onboard_detector/velocity_visualizaton", MarkerArray,
                         self._vel_cb, queue_size=5)

        # 정책 타이머
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._policy_cb)

        rospy.loginfo("[Policy] Corridor policy node started.")

    # ──────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        # quaternion → yaw
        siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        with self._lock:
            self.policy.update_robot_pose(np.array([p.x, p.y, p.z]), yaw)
            self.policy.update_robot_vel(vx, vy)

    def _tracked_cb(self, msg):
        """tracked_bboxes MarkerArray → Box 리스트로 변환 후 policy 주입."""
        bboxes = []
        for m in msg.markers:
            if m.type != _BOX_MARKER_TYPE:
                continue
            b = _extract_box_from_marker(m)
            # 저장된 속도 적용
            vel = self._vel_map.get(m.id, (0.0, 0.0))
            b.Vx, b.Vy = vel
            bboxes.append(b)
        with self._lock:
            self.policy.update_tracked_bboxes(bboxes)

    def _dynamic_cb(self, msg):
        bboxes = []
        for m in msg.markers:
            if m.type != _BOX_MARKER_TYPE:
                continue
            bboxes.append(_extract_box_from_marker(m))
        with self._lock:
            self.policy.update_dynamic_bboxes(bboxes)

    def _vel_cb(self, msg):
        """velocity_visualizaton MarkerArray TEXT marker에서 Vx, Vy 파싱."""
        vel_map = {}
        for m in msg.markers:
            # text: "Vx=0.12, Vy=-0.05, |V|=0.13"
            try:
                parts = m.text.split(",")
                vx = float(parts[0].split("=")[1])
                vy = float(parts[1].split("=")[1])
                vel_map[m.id] = (vx, vy)
            except Exception:
                pass
        self._vel_map = vel_map

    def _policy_cb(self, event):
        with self._lock:
            twist = self.policy.step()
        state = self.policy.state
        self._pub_cmd.publish(twist)
        rospy.loginfo_throttle(1.0,
            f"[Policy] state={state:12s}  "
            f"vx={twist.linear.x:.2f}  vy={twist.linear.y:.2f}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = CorridorPolicyNode()
    node.spin()
