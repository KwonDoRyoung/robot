"""
FILE: kalman_filter.py
--------------------------------------
Python port of kalmanFilter.h/.cpp
Kalman filter velocity estimator
"""

import numpy as np


class KalmanFilter:
    """
    C++ kalman_filter 클래스의 Python 포팅.
    모든 행렬은 np.ndarray (float64).
    states는 항상 shape (N, 1) 컬럼 벡터.
    """

    def __init__(self):
        self.is_initialized = False
        self.states = None
        self.A = None  # 상태 전이 행렬
        self.B = None  # 입력 행렬
        self.H = None  # 관측 행렬
        self.P = None  # 불확실성 (공분산)
        self.Q = None  # 프로세스 노이즈
        self.R = None  # 관측 노이즈

    def setup(self,
              states: np.ndarray,
              A: np.ndarray,
              B: np.ndarray,
              H: np.ndarray,
              P: np.ndarray,
              Q: np.ndarray,
              R: np.ndarray):
        """필터 초기화 — C++ setup() 동일"""
        self.states = states.copy().astype(np.float64)
        self.A = A.copy().astype(np.float64)
        self.B = B.copy().astype(np.float64)
        self.H = H.copy().astype(np.float64)
        self.P = P.copy().astype(np.float64)
        self.Q = Q.copy().astype(np.float64)
        self.R = R.copy().astype(np.float64)
        self.is_initialized = True

    def set_A(self, A: np.ndarray):
        """상태 전이 행렬 교체 (샘플링 시간 변경 시) — C++ setA() 동일"""
        self.A = A.copy().astype(np.float64)

    def estimate(self, z: np.ndarray, u: np.ndarray):
        """
        예측 → 업데이트 — C++ estimate() 동일
        z: 관측 벡터 (M, 1)
        u: 입력 벡터 (입력 행렬 B의 열 수 × 1)
        """
        # predict
        self.states = self.A @ self.states + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # update
        S = self.R + self.H @ self.P @ self.H.T          # innovation matrix
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.states = self.states + K @ (z - self.H @ self.states)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def output(self, state_index: int) -> float:
        """state_index 번째 상태값 반환 — C++ output() 동일"""
        if self.is_initialized:
            return float(self.states[state_index, 0])
        return 0.0
