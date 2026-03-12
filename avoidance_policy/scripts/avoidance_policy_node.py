#!/usr/bin/env python3
"""
avoidance_policy.py  (twist_mux 연동 버전)
-------------------------------------------
/onboard_detector/dynamic_bboxes 토픽을 구독하여
탐지된 동적 객체별 위치(position)와 속도(velocity)를 추정하고,
상태 머신 기반으로 회피 정책 명령을 /cmd_vel_policy 로 발행한다.

twist_mux 가 /cmd_vel_nav (SLAM, priority 10) 과
/cmd_vel_policy (회피, priority 20) 를 우선순위로 선택하여
/cmd_vel 로 출력한다.

토픽 흐름:
  SLAM Nav → cmd_vel_move → safety_manager → cmd_vel_nav (priority 10) ─┐
                                                                         ├─ twist_mux → cmd_vel → Robot
  onboard_detector → dynamic_bboxes → avoidance_policy → cmd_vel_policy (priority 20) ─┘

  - 위협 없음: avoidance_policy 미발행 → twist_mux 가 cmd_vel_nav 선택
  - 위협 감지: avoidance_policy 가 cmd_vel_policy 발행 → twist_mux 가 자동 override
  - POLICY_DURATION 종료 후 발행 중단 → twist_mux timeout 후 cmd_vel_nav 자동 복귀

속도 추정 방법:
  - 동일 id의 마커에 대해 연속 두 프레임의 position 차이를 시간으로 나눔
  - id가 사라졌다가 재등장하면 해당 id의 이전 데이터를 초기화

상태 머신 (vel_x < 0 전제):
  Zone    vel_x  vel_y  상태                  정책
  LEFT      -      0    L_APPROACH_STRAIGHT   AVOIDANCE_R (turn_right)
  LEFT      -      -    L_APPROACH_CENTER     STOP
  LEFT      -      +    L_APPROACH_OUTWARD    SAFE (미발행)
  CENTER    -      0    C_APPROACH            STOP
  CENTER    -      +    C_APPROACH_VEER_LEFT  AVOIDANCE_R (turn_right)
  CENTER    -      -    C_APPROACH_VEER_RIGHT AVOIDANCE_L (turn_left)
  RIGHT     -      0    R_APPROACH_STRAIGHT   AVOIDANCE_L (turn_left)
  RIGHT     -      +    R_APPROACH_CENTER     STOP
  RIGHT     -      -    R_APPROACH_OUTWARD    SAFE (미발행)
  (vel_x >= 0)          SAFE (미발행)
"""

import math
import rospy
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist


class AvoidancePolicy:

    # ── 튜닝 파라미터 ────────────────────────────────────────────────────────
    # XY 이동 거리가 이 값(m) 미만이면 위치 변화 없음으로 판단 (deadband 필터)
    MOVE_THRESH = 0.3

    # pos_y 기준 Zone 경계 (m): |pos_y| <= ZONE_THRESH → CENTER
    ZONE_THRESH = 0.2

    # 이 값(m/s) 이하이면 해당 축 속도를 0으로 판단
    VEL_ZERO_THRESH = 0.2

    # 같은 id가 이 시간(초) 이상 수신되지 않으면 트래킹 데이터 초기화
    TRACK_TIMEOUT = 1.0

    # 정책을 수행할 pos_x 범위 (m): 이 범위 밖의 객체는 무시
    POLICY_X_MIN = 1.0
    POLICY_X_MAX = 4.0

    # 동적 객체 수가 이 값을 초과하면 무조건 STOP (단일 객체 정책만 유효)
    MAX_OBJECTS_FOR_POLICY = 1

    # stop/turn 정책이 최초 트리거된 후 이 시간(초) 동안 유지
    POLICY_DURATION = 1.75     # 정책 총 시간 (초): Phase1(2s) + Phase2(2s)
    PHASE1_DURATION = 0.75     # Phase1 회피 시간 (초)
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        rospy.init_node("avoidance_policy", anonymous=False)

        # id → {"pos": (x,y,z), "stamp": rospy.Time}
        self._prev = {}

        # 현재 진행 중인 정책: "STOP" / "TURN_LEFT" / "TURN_RIGHT" / None
        self._policy_active = None
        self._policy_start  = rospy.Time(0)

        # twist_mux 고우선순위 토픽으로 발행 (위협 감지 시에만)
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel_policy", Twist, queue_size=1)

        rospy.Subscriber(
            "/onboard_detector/dynamic_bboxes",
            MarkerArray,
            self._bbox_cb,
            queue_size=1,
        )

        rospy.loginfo("[avoidance_policy] Started (twist_mux mode).")
        rospy.spin()

    # ── Zone 분류 ────────────────────────────────────────────────────────────

    def _classify_zone(self, pos_y: float) -> str:
        """pos_y 값으로 객체가 속한 Zone 반환."""
        if pos_y > self.ZONE_THRESH:
            return "LEFT"
        elif pos_y < -self.ZONE_THRESH:
            return "RIGHT"
        else:
            return "CENTER"

    # ── 속도 부호 분류 ───────────────────────────────────────────────────────

    def _vel_sign(self, v: float) -> int:
        """속도 값의 부호 반환: -1 / 0 / +1."""
        if v > self.VEL_ZERO_THRESH:
            return 1
        elif v < -self.VEL_ZERO_THRESH:
            return -1
        else:
            return 0

    # ── 상태 결정 ────────────────────────────────────────────────────────────

    def _classify_state(self, zone: str, sign_vx: int, sign_vy: int) -> str:
        """(zone, sign_vx, sign_vy) 조합으로 상태 문자열 반환."""
        if sign_vx >= 0:
            return "SAFE"

        if zone == "LEFT":
            if sign_vy == 0:
                return "L_APPROACH_STRAIGHT"
            elif sign_vy == -1:
                return "L_APPROACH_CENTER"
            else:
                return "L_APPROACH_OUTWARD"

        elif zone == "CENTER":
            if sign_vy == 0:
                return "C_APPROACH"
            elif sign_vy == 1:
                return "C_APPROACH_VEER_LEFT"
            else:
                return "C_APPROACH_VEER_RIGHT"

        else:  # zone == "RIGHT"
            if sign_vy == 0:
                return "R_APPROACH_STRAIGHT"
            elif sign_vy == 1:
                return "R_APPROACH_CENTER"
            else:
                return "R_APPROACH_OUTWARD"

    # ── 정책 함수 ────────────────────────────────────────────────────────────

    def _publish_policy(self, twist: Twist, name: str,
                        state: str, zone: str, sign_vx: int, sign_vy: int):
        """정책 Twist 발행 + 타이머 시작 (최초 1회)."""
        if self._policy_active is None:
            self._policy_active = name
            self._policy_start  = rospy.Time.now()
        self._cmd_vel_pub.publish(twist)
        rospy.loginfo(
            f"[policy] {name} | state={state} | zone={zone} "
            f"| vx_sign={sign_vx:+d}  vy_sign={sign_vy:+d}"
        )

    def stop_policy(self, state: str, zone: str, sign_vx: int, sign_vy: int):
        """linear=0.0, angular=0.0 으로 정지."""
        twist = Twist()
        self._publish_policy(twist, "STOP", state, zone, sign_vx, sign_vy)

    def turn_left_policy(self, state: str, zone: str, sign_vx: int, sign_vy: int):
        """Phase1: 좌회피, Phase2: 우복귀."""
        twist = Twist()
        twist.linear.x = 0.36
        twist.angular.z = 0.72
        self._publish_policy(twist, "TURN_LEFT", state, zone, sign_vx, sign_vy)

    def turn_right_policy(self, state: str, zone: str, sign_vx: int, sign_vy: int):
        """Phase1: 우회피, Phase2: 좌복귀."""
        twist = Twist()
        twist.linear.x = 0.36
        twist.angular.z = -0.72
        self._publish_policy(twist, "TURN_RIGHT", state, zone, sign_vx, sign_vy)

    # ── 정책 실행 ────────────────────────────────────────────────────────────

    def _execute_policy(self, state: str, obj_id: int,
                        px: float, py: float,
                        vx: float, vy: float,
                        zone: str, sign_vx: int, sign_vy: int):
        """상태별 정책 실행. SAFE/OUTWARD → 미발행 (twist_mux가 SLAM 유지)."""

        if state == "L_APPROACH_STRAIGHT":
            self.turn_right_policy(state, zone, sign_vx, sign_vy)

        elif state == "L_APPROACH_CENTER":
            self.turn_right_policy(state, zone, sign_vx, sign_vy)

        elif state == "C_APPROACH":
            self.stop_policy(state, zone, sign_vx, sign_vy)

        elif state == "C_APPROACH_VEER_LEFT":
            self.turn_right_policy(state, zone, sign_vx, sign_vy)

        elif state == "C_APPROACH_VEER_RIGHT":
            self.turn_left_policy(state, zone, sign_vx, sign_vy)

        elif state == "R_APPROACH_STRAIGHT":
            self.turn_left_policy(state, zone, sign_vx, sign_vy)

        elif state == "R_APPROACH_CENTER":
            self.turn_left_policy(state, zone, sign_vx, sign_vy)

        # SAFE, L_APPROACH_OUTWARD, R_APPROACH_OUTWARD → 미발행

    # ── stale id 정리 ─────────────────────────────────────────────────────

    def _cleanup_stale(self, now, current_ids: set):
        """TRACK_TIMEOUT 초과한 id를 _prev에서 제거."""
        stale_ids = [
            mid for mid, data in self._prev.items()
            if mid not in current_ids
            and (now - data["stamp"]).to_sec() > self.TRACK_TIMEOUT
        ]
        for mid in stale_ids:
            del self._prev[mid]
            rospy.loginfo(f"[obj {mid:2d}] LOST (timeout)")

    # ── Subscriber 콜백 ──────────────────────────────────────────────────────

    def _bbox_cb(self, msg: MarkerArray):
        now = rospy.Time.now()

        # ── 진행 중인 정책 유지 (POLICY_DURATION 동안 계속 발행) ────────────
        if self._policy_active is not None:
            elapsed = (now - self._policy_start).to_sec()
            if elapsed < self.POLICY_DURATION:
                twist = Twist()
                phase1 = elapsed < self.PHASE1_DURATION
                if self._policy_active == "TURN_LEFT":
                    if phase1:
                        twist.linear.x = 0.36
                        twist.angular.z = 0.72
                    else:
                        twist.linear.x = 0.225
                        twist.angular.z = -0.82
                elif self._policy_active == "TURN_RIGHT":
                    if phase1:
                        twist.linear.x = 0.36
                        twist.angular.z = -0.72
                    else:
                        twist.linear.x = 0.225
                        twist.angular.z = 0.82
                # STOP: twist 기본값 0.0
                self._cmd_vel_pub.publish(twist)
                return
            else:
                rospy.loginfo(
                    f"[policy] {self._policy_active} 종료 ({elapsed:.1f}s) → SLAM 복귀"
                )
                self._policy_active = None
                self._prev.clear()  # 트래킹 데이터 초기화 → 다음 에피소드 새로 시작
        # ────────────────────────────────────────────────────────────────────

        # ── stale id 정리 (마커 비어있어도 실행) ──────────────────────────
        self._cleanup_stale(now, set(m.id for m in msg.markers))

        # 동적 객체 없음 → 미발행 (twist_mux가 cmd_vel_nav 선택)
        if len(msg.markers) == 0:
            return

        # 동적 객체 수가 MAX_OBJECTS_FOR_POLICY 초과 시 무조건 STOP
        if len(msg.markers) > self.MAX_OBJECTS_FOR_POLICY:
            rospy.loginfo(
                f"[policy] MULTI_OBJECT ({len(msg.markers)} objs "
                f"> {self.MAX_OBJECTS_FOR_POLICY}) → STOP"
            )
            self.stop_policy("MULTI_OBJECT", "NONE", 0, 0)
            return

        for marker in msg.markers:
            mid = marker.id

            px = marker.pose.position.x
            py = marker.pose.position.y
            pz = marker.pose.position.z

            prev = self._prev.get(mid)

            if prev is None:
                self._prev[mid] = {"pos": (px, py, pz), "stamp": now}
                rospy.loginfo(f"[obj {mid:2d}] NEW | pos=({px:.3f}, {py:.3f}, {pz:.3f})")
                continue

            dt = (now - prev["stamp"]).to_sec()
            if dt <= 0.0:
                continue

            ox, oy, oz = prev["pos"]

            # Deadband 필터: XY 이동 거리가 MOVE_THRESH 미만이면 스킵
            dist_xy = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            if dist_xy < self.MOVE_THRESH:
                # stamp만 갱신하여 다음 프레임에서 TRACK_TIMEOUT 방지
                self._prev[mid]["stamp"] = now
                continue

            # pos_x 범위 필터: POLICY_X_MIN ~ POLICY_X_MAX 밖이면 무시
            if not (self.POLICY_X_MIN <= px <= self.POLICY_X_MAX):
                self._prev[mid] = {"pos": (px, py, pz), "stamp": now}
                continue

            # 속도 추정
            vx = (px - ox) / dt
            vy = (py - oy) / dt
            vz = (pz - oz) / dt
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            # Zone 및 상태 결정
            zone = self._classify_zone(py)
            sign_vx = self._vel_sign(vx)
            sign_vy = self._vel_sign(vy)
            state = self._classify_state(zone, sign_vx, sign_vy)

            rospy.loginfo(
                f"[obj {mid:2d}] "
                f"zone={zone:6s} | "
                f"pos=({px:6.3f}, {py:6.3f}) | "
                f"vel=({vx:6.3f}, {vy:6.3f}) |v|={speed:.3f} m/s | "
                f"state={state}"
            )

            # 정책 실행 (SAFE/OUTWARD는 미발행 → twist_mux가 SLAM 유지)
            self._execute_policy(state, mid, px, py, vx, vy, zone, sign_vx, sign_vy)

            # 현재 데이터 저장
            self._prev[mid] = {"pos": (px, py, pz), "stamp": now}



if __name__ == "__main__":
    try:
        AvoidancePolicy()
    except rospy.ROSInterruptException:
        pass
