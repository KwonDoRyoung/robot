"""
FILE: utils.py
--------------------------
Python port of utils.h
Function utils for detectors
"""

import math
import copy
import numpy as np
from dataclasses import dataclass, field

PI_const = 3.1415926


@dataclass
class Box3D:
    """Python equivalent of struct box3D in utils.h"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    x_width: float = 0.0
    y_width: float = 0.0
    z_width: float = 0.0
    id: float = 0.0
    Vx: float = 0.0
    Vy: float = 0.0
    Vz: float = 0.0
    Ax: float = 0.0
    Ay: float = 0.0
    Az: float = 0.0
    is_human: bool = False          # YOLO로 동적 감지 여부
    is_dynamic: bool = False        # YOLO 또는 분류 콜백으로 동적 감지 여부
    fix_size: bool = False          # 크기 고정 플래그
    is_dynamic_candidate: bool = False
    is_estimated: bool = False

    def copy(self):
        return copy.deepcopy(self)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float):
    """
    geometry_msgs/Quaternion 메시지 반환 (C++ quaternion_from_rpy 동일)
    yaw > PI 이면 -2*PI 보정
    """
    from geometry_msgs.msg import Quaternion
    from tf.transformations import quaternion_from_euler

    if yaw > PI_const:
        yaw = yaw - 2 * PI_const

    q = quaternion_from_euler(roll, pitch, yaw)  # [x, y, z, w]
    msg = Quaternion()
    msg.x = q[0]
    msg.y = q[1]
    msg.z = q[2]
    msg.w = q[3]
    return msg


def rpy_from_quaternion(quat) -> float:
    """
    geometry_msgs/Quaternion → yaw (float) 반환
    C++ rpy_from_quaternion(quat) 단일 반환값 버전
    """
    from tf.transformations import euler_from_quaternion
    _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    return yaw


def rpy_from_quaternion_full(quat):
    """
    geometry_msgs/Quaternion → (roll, pitch, yaw) 반환
    C++ rpy_from_quaternion(quat, roll, pitch, yaw) 3인수 버전
    """
    from tf.transformations import euler_from_quaternion
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(quat) -> np.ndarray:
    """
    geometry_msgs/Quaternion → 3×3 회전 행렬 (numpy)
    """
    from tf.transformations import quaternion_matrix
    mat4 = quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    return mat4[:3, :3]


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 3D 벡터 사이 각도 (라디안) — C++ angleBetweenVectors 동일
    atan2(|a×b|, a·b)
    """
    cross_norm = np.linalg.norm(np.cross(a, b))
    dot = np.dot(a, b)
    return math.atan2(cross_norm, dot)


def compute_center(points) -> np.ndarray:
    """
    3D 점 목록의 중심 계산 — C++ computeCenter 동일
    points: list of np.ndarray shape (3,) 또는 Nx3 ndarray
    """
    if len(points) == 0:
        return np.zeros(3)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        return arr.copy()
    return arr.mean(axis=0)


def compute_std(points, center: np.ndarray) -> np.ndarray:
    """
    3D 점 목록의 표준편차 계산 — C++ computeStd 동일
    모집단 분산 기준 (÷N, not N-1)
    """
    if len(points) == 0:
        return np.zeros(3)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        diff = arr - center
        return np.sqrt(diff * diff)
    diff = arr - center
    variance = np.mean(diff * diff, axis=0)
    return np.sqrt(variance)
