#!/usr/bin/env python3
"""
corridor_policy.py
------------------
일자 복도 주행 정책 (시나리오 1 & 2)

시나리오 1: 고정 장애물만 있음
  - 위험 구간 진입 시 서행(slow_speed)으로 통과

시나리오 2: 사람이 로봇 방향으로 접근
  - 접근 객체의 로봇 기준 좌/우 판단
  - 반대쪽 벽 방향으로 횡이동(lateral)
  - 통과 후 정상 속도 복귀
"""

import math
import numpy as np
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

from onboard_detector_python.utils import Box3D


# ──────────────────────────────────────────────────
# 속도 상수 (m/s)
# ──────────────────────────────────────────────────
SPEED_NORMAL  = 0.3   # 정상 전진 속도
SPEED_SLOW    = 0.1   # 위험 구간 서행
SPEED_STOP    = 0.0   # 정지
LATERAL_SPEED = 0.2   # 횡이동 속도 (±)
YAW_RATE      = 0.0   # 복도 직진 가정 (회전 없음)

# 접근 판정 임계값
APPROACH_VEL_THRESH   = 0.15   # m/s, 이 이상 로봇 방향으로 오면 "접근 중"
APPROACH_DIST_THRESH  = 3.0    # m, 이 거리 이내에서만 판정

# 위험 구간: 로봇이 이 Y 범위에 있을 때 서행 (localization 기준)
# 실제 환경에 맞게 yaml 파라미터로 오버라이드 가능
DANGER_ZONE_DEFAULT = [(-50.0, 50.0)]  # [(y_min, y_max), ...] world frame


class CorridorPolicy:
    """
    상태 머신 기반 복도 주행 정책.

    상태:
      NORMAL     : 정상 전진
      DANGER     : 위험 구간 서행 (시나리오 1)
      AVOID_LEFT : 왼쪽으로 횡이동 회피 (시나리오 2)
      AVOID_RIGHT: 오른쪽으로 횡이동 회피 (시나리오 2)
      STOP       : 긴급 정지
    """

    STATE_NORMAL      = "NORMAL"
    STATE_DANGER      = "DANGER"
    STATE_AVOID_LEFT  = "AVOID_LEFT"
    STATE_AVOID_RIGHT = "AVOID_RIGHT"
    STATE_STOP        = "STOP"

    def __init__(self):
        self.state = self.STATE_NORMAL

        # 로봇 pose
        self.robot_pos = np.zeros(3)        # world frame
        self.robot_yaw = 0.0               # rad
        self.robot_vel = np.zeros(2)       # vx, vy body frame

        # 감지 데이터
        self.tracked_bboxes  = []          # List[Box3D] from tracker
        self.dynamic_bboxes  = []          # List[Box3D] classified as dynamic

        # 위험 구간 목록 [(y_min, y_max)]
        self.danger_zones = DANGER_ZONE_DEFAULT

        # 회피 완료 카운터 (상태 유지 프레임 수)
        self._avoid_hold = 0
        self.AVOID_HOLD_FRAMES = 20   # 1초(20Hz) 동안 회피 유지

    # ──────────────────────────────────────────────────
    # 외부에서 데이터 주입
    # ──────────────────────────────────────────────────

    def update_robot_pose(self, pos: np.ndarray, yaw: float):
        self.robot_pos = pos
        self.robot_yaw = yaw

    def update_robot_vel(self, vx: float, vy: float):
        self.robot_vel = np.array([vx, vy])

    def update_tracked_bboxes(self, bboxes):
        self.tracked_bboxes = bboxes

    def update_dynamic_bboxes(self, bboxes):
        self.dynamic_bboxes = bboxes

    def set_danger_zones(self, zones):
        """zones: [(y_min, y_max), ...]  world frame y 기준"""
        self.danger_zones = zones

    # ──────────────────────────────────────────────────
    # 메인 스텝 (타이머에서 호출)
    # ──────────────────────────────────────────────────

    def step(self) -> Twist:
        """현재 상태를 평가하고 cmd_vel Twist를 반환한다."""
        twist = Twist()

        # 1. 접근 객체 분석 (시나리오 2 우선 판정)
        approach_result = self._check_approaching_object()

        # 2. 위험 구간 판정 (시나리오 1)
        in_danger = self._in_danger_zone()

        # ── 상태 전이 ──
        if approach_result is not None:
            # 동적 객체 접근 → 회피 모드
            side = approach_result  # "left" or "right"
            if side == "right":
                self.state = self.STATE_AVOID_LEFT   # 오른쪽에서 오면 왼쪽으로
            else:
                self.state = self.STATE_AVOID_RIGHT  # 왼쪽에서 오면 오른쪽으로
            self._avoid_hold = self.AVOID_HOLD_FRAMES

        elif self._avoid_hold > 0:
            # 회피 유지 중
            self._avoid_hold -= 1
            if self._avoid_hold == 0:
                self.state = self.STATE_NORMAL

        elif in_danger:
            self.state = self.STATE_DANGER

        else:
            self.state = self.STATE_NORMAL

        # ── 상태별 cmd_vel 생성 ──
        if self.state == self.STATE_NORMAL:
            twist.linear.x  = SPEED_NORMAL
            twist.linear.y  = 0.0
            twist.angular.z = YAW_RATE

        elif self.state == self.STATE_DANGER:
            twist.linear.x  = SPEED_SLOW
            twist.linear.y  = 0.0
            twist.angular.z = YAW_RATE

        elif self.state == self.STATE_AVOID_LEFT:
            twist.linear.x  = SPEED_SLOW
            twist.linear.y  = LATERAL_SPEED        # +y = 왼쪽 (ROS convention)
            twist.angular.z = YAW_RATE

        elif self.state == self.STATE_AVOID_RIGHT:
            twist.linear.x  = SPEED_SLOW
            twist.linear.y  = -LATERAL_SPEED       # -y = 오른쪽
            twist.angular.z = YAW_RATE

        elif self.state == self.STATE_STOP:
            pass  # 모든 속도 0

        return twist

    # ──────────────────────────────────────────────────
    # 내부 판정 함수
    # ──────────────────────────────────────────────────

    def _in_danger_zone(self) -> bool:
        """로봇의 world-frame y 좌표가 위험 구간 내에 있는지 확인."""
        y = self.robot_pos[1]
        for (y_min, y_max) in self.danger_zones:
            if y_min <= y <= y_max:
                return True
        return False

    def _check_approaching_object(self):
        """
        tracked_bboxes 중 로봇 방향으로 접근하는 객체가 있으면
        그 객체가 로봇 기준 'left' 또는 'right' 반환.
        없으면 None.
        """
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)

        for bbox in self.tracked_bboxes:
            obj_pos = np.array([bbox.x, bbox.y])
            robot_pos_2d = self.robot_pos[:2]

            # 거리 체크
            dist = np.linalg.norm(obj_pos - robot_pos_2d)
            if dist > APPROACH_DIST_THRESH:
                continue

            # 객체 속도 (Vx, Vy world frame)
            obj_vel = np.array([bbox.Vx, bbox.Vy])
            if np.linalg.norm(obj_vel) < 0.05:
                continue

            # 로봇→객체 방향 벡터
            to_obj = obj_pos - robot_pos_2d
            to_obj_norm = np.linalg.norm(to_obj)
            if to_obj_norm < 1e-6:
                continue
            to_obj_unit = to_obj / to_obj_norm

            # 객체 속도를 로봇→객체 방향으로 투영
            # 음수 = 로봇 방향으로 접근
            approach_vel = -np.dot(obj_vel, to_obj_unit)
            if approach_vel < APPROACH_VEL_THRESH:
                continue

            # 로봇 기준 좌/우 판정 (cross product)
            # robot_forward = (cos_yaw, sin_yaw)
            robot_fwd = np.array([cos_yaw, sin_yaw])
            cross = robot_fwd[0] * to_obj[1] - robot_fwd[1] * to_obj[0]
            side = "left" if cross > 0 else "right"
            return side

        return None
