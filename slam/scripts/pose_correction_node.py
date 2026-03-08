#!/usr/bin/env python3
"""
pose_correction_node.py

Detects and corrects sudden position/rotation jumps in localization.
Provides smooth pose output by filtering discontinuities.

Fixes:
  - 위치 보정: rejects pose jumps > threshold, triggers re-localization
  - 노이즈 문제: exponential moving average on pose
  - 회전 보정: handles yaw wrap-around, rejects large rotation jumps
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def quat_to_yaw(q):
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def yaw_to_quat(yaw):
    return quaternion_from_euler(0.0, 0.0, yaw)


def angle_diff(a, b):
    """Smallest signed angle difference (handles wrap-around)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


class PoseCorrectionNode:
    def __init__(self):
        rospy.init_node("pose_correction_node")

        # Thresholds
        self.max_pos_jump   = rospy.get_param("~max_position_jump_m",  1.5)   # meters
        self.max_yaw_jump   = rospy.get_param("~max_yaw_jump_deg", 30.0) * np.pi / 180.0
        self.ema_alpha      = rospy.get_param("~ema_alpha", 0.7)   # 0=heavy smooth, 1=no smooth
        self.reloc_timeout  = rospy.get_param("~relocalization_timeout_s", 5.0)

        # State
        self.last_pose = None       # (x, y, yaw)
        self.smoothed_pose = None   # (x, y, yaw)
        self.jump_count = 0
        self.last_valid_time = None
        self.relocalization_triggered = False

        # Subscribe to HDL localization odom
        self.sub_odom = rospy.Subscriber(
            "/odom_3d", Odometry, self.odom_callback, queue_size=5)

        # Publish corrected pose
        self.pub_pose = rospy.Publisher(
            "/odom_3d_corrected", Odometry, queue_size=5)
        self.pub_jump = rospy.Publisher(
            "/slam/pose_jump_detected", Bool, queue_size=1)
        self.pub_relocalize = rospy.Publisher(
            "/slam/trigger_relocalization", Bool, queue_size=1)

        rospy.loginfo(
            f"[PoseCorrection] max_pos={self.max_pos_jump:.1f}m "
            f"max_yaw={np.degrees(self.max_yaw_jump):.1f}deg "
            f"ema={self.ema_alpha:.2f}"
        )

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        x   = pos.x
        y   = pos.y
        yaw = quat_to_yaw(ori)

        if self.last_pose is None:
            # Initialize
            self.last_pose    = (x, y, yaw)
            self.smoothed_pose = (x, y, yaw)
            self.last_valid_time = msg.header.stamp.to_sec()
            self.pub_pose.publish(msg)
            return

        lx, ly, lyaw = self.last_pose
        sx, sy, syaw = self.smoothed_pose

        # --- Jump detection ---
        pos_diff = np.hypot(x - lx, y - ly)
        yaw_diff = abs(angle_diff(yaw, lyaw))

        dt = msg.header.stamp.to_sec() - self.last_valid_time
        is_jump = (pos_diff > self.max_pos_jump) or (yaw_diff > self.max_yaw_jump)

        if is_jump:
            self.jump_count += 1
            rospy.logwarn(
                f"[PoseCorrection] Jump #{self.jump_count}: "
                f"pos={pos_diff:.2f}m yaw={np.degrees(yaw_diff):.1f}deg - rejected"
            )
            self.pub_jump.publish(Bool(data=True))

            # If jumps persist → trigger re-localization
            if self.jump_count >= 3 and dt > self.reloc_timeout:
                if not self.relocalization_triggered:
                    rospy.logwarn("[PoseCorrection] Persistent jumps → triggering re-localization!")
                    self.pub_relocalize.publish(Bool(data=True))
                    self.relocalization_triggered = True
                    self.jump_count = 0

            # Publish last known good smoothed pose (don't update)
            out = self._make_odom(msg, sx, sy, syaw)
            self.pub_pose.publish(out)
            return

        # --- Reset jump counter on valid pose ---
        self.jump_count = 0
        self.relocalization_triggered = False
        self.last_valid_time = msg.header.stamp.to_sec()
        self.pub_jump.publish(Bool(data=False))

        # --- EMA smoothing ---
        a = self.ema_alpha
        nx = a * x   + (1 - a) * sx
        ny = a * y   + (1 - a) * sy
        # Yaw wrap-aware smoothing
        nyaw = syaw + a * angle_diff(yaw, syaw)

        self.last_pose    = (x, y, yaw)
        self.smoothed_pose = (nx, ny, nyaw)

        out = self._make_odom(msg, nx, ny, nyaw)
        self.pub_pose.publish(out)

    def _make_odom(self, template: Odometry, x, y, yaw) -> Odometry:
        out = Odometry()
        out.header = template.header
        out.child_frame_id = template.child_frame_id
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = template.pose.pose.position.z
        q = yaw_to_quat(yaw)
        out.pose.pose.orientation.x = q[0]
        out.pose.pose.orientation.y = q[1]
        out.pose.pose.orientation.z = q[2]
        out.pose.pose.orientation.w = q[3]
        out.pose.covariance   = template.pose.covariance
        out.twist             = template.twist
        return out

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = PoseCorrectionNode()
    node.spin()
