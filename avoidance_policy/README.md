# 복도 주행 정책 (Corridor Navigation Policy)

## 파일 구성

| 파일 | 설명 |
|------|------|
| `corridor_policy.py` | 상태 머신 정책 로직 (ROS 비의존) |
| `corridor_policy_node.py` | ROS1 노드 래퍼 |
| `run_policy.launch` | 실행 런치 파일 |

---

## 시나리오

### 시나리오 1 — 고정 장애물, 위험 구간 서행

```
상태: NORMAL → DANGER → NORMAL
```

- `danger_zone_y` 파라미터로 world frame y 좌표 기준 위험 구간 지정
- 진입 시 `slow_speed`(0.1 m/s)로 서행, 통과 후 `normal_speed` 복귀

### 시나리오 2 — 사람 접근 시 횡이동 회피

```
상태: NORMAL → AVOID_LEFT or AVOID_RIGHT → NORMAL
```

- `/onboard_detector/tracked_bboxes` + 속도 정보로 접근 객체 감지
- 접근 방향 판정:
  - 오른쪽에서 접근 → `AVOID_LEFT` (+y 방향 횡이동)
  - 왼쪽에서 접근  → `AVOID_RIGHT` (-y 방향 횡이동)
- `AVOID_HOLD_FRAMES`(기본 20프레임 = 1초) 동안 회피 상태 유지

---

## 상태 머신

```
         진입 판정          회피 완료(1초 후)
NORMAL ──────────► AVOID_LEFT/RIGHT ──────────► NORMAL
   │                                                ▲
   │ danger_zone 진입                               │
   ▼                danger_zone 탈출               │
DANGER ─────────────────────────────────────────────┘
```

---

## 토픽

| 방향 | 토픽 | 타입 |
|------|------|------|
| 구독 | `/odom_3d` | `nav_msgs/Odometry` |
| 구독 | `/onboard_detector/tracked_bboxes` | `visualization_msgs/MarkerArray` |
| 구독 | `/onboard_detector/dynamic_bboxes` | `visualization_msgs/MarkerArray` |
| 구독 | `/onboard_detector/velocity_visualizaton` | `visualization_msgs/MarkerArray` |
| 발행 | `/cmd_vel` | `geometry_msgs/Twist` |

---

## 실행

```bash
# 감지기 먼저 실행
roslaunch onboard_detector_python run_detector.launch

# 정책 노드 실행
cd /home/irop/projects/robot/policy
roslaunch run_policy.launch
```

---

## 파라미터 튜닝

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `normal_speed` | 0.3 | 정상 전진 속도 (m/s) |
| `slow_speed` | 0.1 | 서행 속도 (m/s) |
| `lateral_speed` | 0.2 | 횡이동 속도 (m/s) |
| `approach_vel_thresh` | 0.15 | 접근 판정 최소 속도 (m/s) |
| `approach_dist_thresh` | 3.0 | 접근 판정 최대 거리 (m) |
| `danger_zone_y` | `[5.0, 15.0]` | 위험 구간 y 범위 (world frame) |
| `policy_rate` | 20.0 | 정책 실행 주기 (Hz) |
