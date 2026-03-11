"""
FILE: uv_detector.py
------------------
Python port of uvDetector.h/.cpp
UV-map 기반 깊이 이미지 장애물 감지기
"""

import math
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:
    # numba 없으면 no-op decorator
    # @_njit(cache=True) → _njit(cache=True) 가 decorator 반환 → decorator(fn) = fn
    def _njit(**kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    _NUMBA_OK = False


@_njit(cache=True)
def _extract_bb_scan(u_map, u_min, min_length_line, threshold_line):
    """
    U_map을 행 단위로 스캔해 세그먼트를 찾고 union-find로 연결.
    반환: mask, seg_row, seg_left, seg_right, seg_parent (모두 numpy array)
    """
    rows, cols = u_map.shape
    mask = np.zeros((rows, cols), dtype=np.int32)

    MAX_SEGS = 4096
    seg_row    = np.zeros(MAX_SEGS, dtype=np.int32)
    seg_left   = np.zeros(MAX_SEGS, dtype=np.int32)
    seg_right  = np.zeros(MAX_SEGS, dtype=np.int32)
    seg_parent = np.zeros(MAX_SEGS, dtype=np.int32)
    n = 0

    for row in range(rows):
        sum_line    = 0
        max_line    = 0
        length_line = 0

        col = 0
        while col <= cols:
            val = int(u_map[row, col]) if col < cols else 0

            if col < cols and val >= u_min:
                length_line += 1
                sum_line    += val
                if val > max_line:
                    max_line = val

            if val < u_min or col == cols - 1:
                end_col = col + 1 if col == cols - 1 else col

                if length_line > min_length_line and sum_line > threshold_line * max_line:
                    seg_id = n + 1
                    s = end_col - length_line
                    e = end_col - 1

                    seg_row[n]    = row
                    seg_left[n]   = s
                    seg_right[n]  = e
                    seg_parent[n] = seg_id

                    for c in range(s, e):
                        mask[row, c] = seg_id

                    if row != 0:
                        for c in range(s, min(e, cols)):
                            prev_id = mask[row - 1, c]
                            if prev_id != 0:
                                prev_par = seg_parent[prev_id - 1]
                                curr_par = seg_parent[n]
                                if prev_par < curr_par:
                                    seg_parent[n] = prev_par
                                else:
                                    for b in range(n + 1):
                                        if seg_parent[b] == prev_par:
                                            seg_parent[b] = curr_par
                                break  # 첫 번째 연결만 처리

                    n += 1

                sum_line    = 0
                max_line    = 0
                length_line = 0

            col += 1

    return mask, seg_row[:n], seg_left[:n], seg_right[:n], seg_parent[:n]

from onboard_detector_python.utils import Box3D
from onboard_detector_python.kalman_filter import KalmanFilter


# ─────────────────────────────────────────
# Rect 헬퍼 (cv2.Rect 대체)
# ─────────────────────────────────────────

class Rect:
    """cv::Rect 의 Python 등가물"""
    __slots__ = ('x', 'y', 'width', 'height')

    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    @property
    def tl(self):
        """top-left (x, y)"""
        return (self.x, self.y)

    @property
    def br(self):
        """bottom-right (x+w, y+h)"""
        return (self.x + self.width, self.y + self.height)

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def intersection(self, other: 'Rect') -> 'Rect':
        """두 사각형의 교집합 (C++ & 연산자 동일)"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        if x2 <= x1 or y2 <= y1:
            return Rect(0, 0, 0, 0)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def to_cv_rect(self):
        """(x, y, w, h) 튜플 — cv2.rectangle 에 전달 가능"""
        return (self.x, self.y, self.width, self.height)

    def copy(self) -> 'Rect':
        return Rect(self.x, self.y, self.width, self.height)

    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


# ─────────────────────────────────────────
# UVbox
# ─────────────────────────────────────────

class UVbox:
    """C++ UVbox 클래스 동일"""

    def __init__(self, seg_id: int = 0, row: int = 0, left: int = 0, right: int = 0):
        if seg_id == 0 and row == 0 and left == 0 and right == 0:
            # 기본 생성자
            self.id = 0
            self.toppest_parent_id = 0
            self.bb = Rect(0, 0, 0, 0)
        else:
            self.id = seg_id
            self.toppest_parent_id = seg_id
            # cv::Rect(Point2f(left, row), Point2f(right, row))
            # → x=left, y=row, width=right-left, height=0
            self.bb = Rect(left, row, right - left, 0)


def merge_two_uvbox(father: UVbox, son: UVbox) -> UVbox:
    """C++ merge_two_UVbox 동일"""
    top    = min(father.bb.y, son.bb.y)
    left   = min(father.bb.x, son.bb.x)
    bottom = max(father.bb.y + father.bb.height, son.bb.y + son.bb.height)
    right  = max(father.bb.x + father.bb.width,  son.bb.x + son.bb.width)
    father.bb = Rect(left, top, right - left, bottom - top)
    return father


# ─────────────────────────────────────────
# UVtracker
# ─────────────────────────────────────────

class UVtracker:
    """C++ UVtracker 클래스 동일"""

    def __init__(self):
        self.overlap_threshold: float = 0.4

        self.pre_bb: List[Rect] = []
        self.now_bb: List[Rect] = []
        self.pre_history: List[List] = []   # list of list of (cx, cy) tuples
        self.now_history: List[List] = []
        self.pre_filter: List[KalmanFilter] = []
        self.now_filter: List[KalmanFilter] = []
        self.pre_V: deque = deque()         # deque of deque of np.ndarray (2,1)
        self.now_V: deque = deque()
        self.pre_count: deque = deque()
        self.now_count: deque = deque()
        self.now_bb_D: List[Rect] = []      # depth bounding boxes
        self.now_box_3D: List[Box3D] = []
        self.now_box_3D_history: deque = deque()  # deque of deque of Box3D
        self.pre_box_3D_history: deque = deque()
        self.fixed_box3D: List[Box3D] = []

    def read_bb(self, now_bb: List[Rect], now_bb_D: List[Rect], box_3D: List[Box3D]) -> List[Box3D]:
        """
        C++ read_bb() 동일.
        box_3D 를 인플레이스로 수정하지 않고 수정된 복사본 반환.
        """
        import copy

        # measurement history
        self.pre_history = [list(h) for h in self.now_history]
        self.now_history = [[] for _ in range(len(now_bb))]

        # 3D box history
        self.pre_box_3D_history = deque(
            [deque(dq) for dq in self.now_box_3D_history]
        )
        self.now_box_3D_history = deque(
            [deque() for _ in range(len(now_bb))]
        )

        # kalman filters
        self.pre_filter = list(self.now_filter)
        self.now_filter = [KalmanFilter() for _ in range(len(now_bb))]

        # velocity sum
        self.pre_V = deque(deque(dq) for dq in self.now_V)
        self.now_V = deque(deque() for _ in range(len(now_bb)))

        # bounding boxes
        self.pre_bb = list(self.now_bb)
        self.now_bb = list(now_bb)
        self.now_bb_D = list(now_bb_D)
        self.now_box_3D = copy.deepcopy(box_3D)

        # history size 관리 (>10 이면 pop)
        pre_hist_list = list(self.pre_box_3D_history)
        for i in range(len(pre_hist_list)):
            if len(pre_hist_list[i]) > 10:
                pre_hist_list[i].popleft()
                if i < len(self.pre_history) and len(self.pre_history[i]) > 0:
                    self.pre_history[i].pop(0)
        self.pre_box_3D_history = deque(pre_hist_list)

        return copy.deepcopy(self.now_box_3D)

    def check_status(self, box_3D: List[Box3D]) -> List[Box3D]:
        """C++ check_status() 동일"""
        import copy

        for now_id in range(len(self.now_bb)):
            tracked = False
            nb = self.now_bb[now_id]
            nb_cx = nb.x + 0.5 * nb.width
            nb_cy = nb.y + 0.5 * nb.height

            for pre_id in range(len(self.pre_bb)):
                pb = self.pre_bb[pre_id]
                overlap = nb.intersection(pb)

                pb_cx = pb.x + 0.5 * pb.width
                pb_cy = pb.y + 0.5 * pb.height

                dist = math.sqrt((nb_cx - pb_cx) ** 2 + (nb_cy - pb_cy) ** 2)
                metric = math.sqrt((nb.width + pb.width) ** 2 +
                                   (nb.height + pb.height) ** 2) / 2.0

                nb_area = nb.area()
                pb_area = pb.area()
                overlap_area = overlap.area()

                ratio = 0.0
                if nb_area > 0 and pb_area > 0:
                    ratio = max(overlap_area / float(nb_area),
                                overlap_area / float(pb_area))

                if ratio >= self.overlap_threshold or dist <= metric:
                    tracked = True

                    # inherit history
                    pre_h = self.pre_history[pre_id] if pre_id < len(self.pre_history) else []
                    self.now_history[now_id] = list(pre_h)
                    self.now_history[now_id].append((nb_cx, nb_cy))

                    # 3D box history 상속
                    pre_3dh = (list(self.pre_box_3D_history)[pre_id]
                               if pre_id < len(self.pre_box_3D_history) else deque())
                    self.now_box_3D_history[now_id] = deque(pre_3dh)

                    # depth bbox 경계 체크 후 3D box history에 추가
                    nb_D = self.now_bb_D[now_id] if now_id < len(self.now_bb_D) else Rect()
                    if (nb_D.x > 5 and nb_D.y > 5 and
                            nb_D.x + nb_D.width < 635 and
                            nb_D.y + nb_D.height < 475):
                        self.now_box_3D_history[now_id].append(
                            copy.deepcopy(self.now_box_3D[now_id])
                        )

                    # velocity, filter 상속
                    pre_v = (list(self.pre_V)[pre_id]
                             if pre_id < len(self.pre_V) else deque())
                    self.now_V[now_id] = deque(pre_v)
                    pre_f = self.pre_filter[pre_id] if pre_id < len(self.pre_filter) else KalmanFilter()
                    self.now_filter[now_id] = pre_f

                    break

            if not tracked:
                self.now_history[now_id].append((nb_cx, nb_cy))

                V = np.zeros((2, 1))
                self.now_V[now_id].append(V)

                nb_D = self.now_bb_D[now_id] if now_id < len(self.now_bb_D) else Rect()
                if (nb_D.x > 5 and nb_D.y > 5 and
                        nb_D.x + nb_D.width < 635 and
                        nb_D.y + nb_D.height < 475):
                    self.now_box_3D_history[now_id].append(
                        copy.deepcopy(self.now_box_3D[now_id])
                    )

        return box_3D


# ─────────────────────────────────────────
# UVdetector
# ─────────────────────────────────────────

class UVdetector:
    """C++ UVdetector 클래스 동일"""

    def __init__(self):
        self.row_downsample: int = 4
        self.col_scale: float = 0.5
        self.min_dist: int = 10          # mm
        self.max_dist: int = 8000        # mm
        self.threshold_point: float = 3.0
        self.threshold_line: float = 2.0
        self.min_length_line: int = 6
        self.show_bounding_box_U: bool = True

        # 기본 카메라 내부 파라미터 (나중에 외부에서 설정)
        self.fx: float = 608.08740234375
        self.fy: float = 608.1791381835938
        self.px: float = 317.48284912109375
        self.py: float = 234.11557006835938
        self.depthScale_: float = 1000.0

        self.x0: int = 0
        self.y0: int = 0

        # 내부 데이터
        self.depth: Optional[np.ndarray] = None          # uint16 depth image
        self.depth_show: Optional[np.ndarray] = None
        self.RGB: Optional[np.ndarray] = None
        self.depth_low_res: Optional[np.ndarray] = None
        self.U_map: Optional[np.ndarray] = None
        self.U_map_show: Optional[np.ndarray] = None
        self.bird_view: Optional[np.ndarray] = None

        self.bounding_box_U: List[Rect] = []
        self.bounding_box_B: List[Rect] = []
        self.bounding_box_D: List[Rect] = []
        self.box3Ds: List[Box3D] = []       # 카메라 프레임 3D bbox
        self.box3DsWorld: List[Box3D] = []  # 월드 프레임 3D bbox

        self.testx: int = 0
        self.testy: int = 0
        self.testby: int = 0

        self.tracker = UVtracker()

    # ── 데이터 로드 ──────────────────────────

    def readdepth(self, depth: np.ndarray):
        """C++ readdepth() 동일"""
        self.depth = depth

    def readrgb(self, rgb: np.ndarray):
        """C++ readrgb() 동일 — 720×400으로 리사이즈"""
        self.RGB = cv2.resize(rgb, (720, 400))

    # ── 메인 처리 파이프라인 ─────────────────

    def extract_U_map(self):
        """extract_U_map — numpy 벡터화 버전"""
        new_w = int(self.depth.shape[1] * self.col_scale)
        depth_rescale = cv2.resize(self.depth, (new_w, self.depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        hist_size = self.depth.shape[0] // self.row_downsample
        bin_width = math.ceil((self.max_dist - self.min_dist) / float(hist_size))

        # mm 단위 변환 (벡터화)
        depth_val = (depth_rescale.astype(np.float32) / self.depthScale_ * 1000.0).astype(np.int32)

        valid = (depth_val > self.min_dist) & (depth_val < self.max_dist)
        bin_idx = np.clip((depth_val - self.min_dist) // bin_width, 0, hist_size - 1)

        depth_low_res_temp = np.where(valid, bin_idx, 0).astype(np.uint8)
        self.depth_low_res = depth_low_res_temp

        # U_map: 각 (bin, col) 에 valid 픽셀 수 집계
        # np.add.at는 GIL-holding scalar loop → bincount로 교체 (수배 빠름)
        rows_arr, cols_arr = np.where(valid)
        bins_arr = bin_idx[rows_arr, cols_arr]
        flat_idx = bins_arr * depth_rescale.shape[1] + cols_arr
        u_map_flat = np.bincount(flat_idx, minlength=hist_size * depth_rescale.shape[1])
        u_map = u_map_flat.reshape(hist_size, depth_rescale.shape[1]).astype(np.int32)
        self.U_map = np.clip(u_map, 0, 255).astype(np.uint8)

        # U-map 평활화
        self.U_map = cv2.GaussianBlur(self.U_map, (5, 9), 10, 10)

    def extract_bb(self):
        """C++ extract_bb() 완전 동일.
        Numba 있으면 JIT 가속, 없으면 OpenCV connectedComponents 경로 사용.
        """
        if _NUMBA_OK:
            self._extract_bb_numba()
        else:
            self._extract_bb_opencv()

    def _extract_bb_numba(self):
        """Numba JIT 경로."""
        u_min = int(self.threshold_point * self.row_downsample)
        _, seg_rows, seg_lefts, seg_rights, seg_parents = _extract_bb_scan(
            self.U_map, u_min, self.min_length_line, float(self.threshold_line)
        )
        n = len(seg_rows)
        self.bounding_box_U = []
        for b in range(n):
            if seg_parents[b] != b + 1:
                continue
            top   = int(seg_rows[b]);  bot   = int(seg_rows[b])
            left  = int(seg_lefts[b]); right = int(seg_rights[b])
            for s in range(b + 1, n):
                if seg_parents[s] == b + 1:
                    top   = min(top,   int(seg_rows[s]))
                    left  = min(left,  int(seg_lefts[s]))
                    bot   = max(bot,   int(seg_rows[s]))
                    right = max(right, int(seg_rights[s]))
            bb = Rect(left, top, right - left, bot - top)
            if bb.area() >= 25:
                self.bounding_box_U.append(bb)

    def _extract_bb_opencv(self):
        """OpenCV connectedComponents 경로 (Numba 없을 때 fallback).

        전략:
          1. U_map >= u_min 으로 binary mask 생성
          2. 행별로 numpy diff를 써서 유효 run (길이 > min_length_line,
             품질 조건 sum > threshold_line * max) 만 남긴 filtered mask 생성
          3. cv2.connectedComponentsWithStats (C++ 백엔드) 로 bbox 추출
        """
        u_min          = int(self.threshold_point * self.row_downsample)
        min_len        = self.min_length_line
        thresh_line    = self.threshold_line
        rows, cols     = self.U_map.shape

        # Step 1: threshold → binary
        binary = (self.U_map >= u_min).astype(np.uint8)

        # Step 2: run-quality 필터 (행 루프 120회 정도, 내부는 numpy)
        filtered = np.zeros_like(binary)
        for row in range(rows):
            row_bin = binary[row]
            if not row_bin.any():
                continue
            # run 경계 찾기: diff([0, row_bin, 0])
            padded = np.empty(cols + 2, dtype=np.int8)
            padded[0] = 0; padded[1:-1] = row_bin; padded[-1] = 0
            d = np.diff(padded)
            starts = np.where(d == 1)[0]   # run 시작 col
            ends   = np.where(d == -1)[0]  # run 끝 col (exclusive)
            for s, e in zip(starts, ends):
                length = int(e - s)
                if length <= min_len:
                    continue
                run_vals = self.U_map[row, s:e]
                mx = int(run_vals.max())
                if mx == 0:
                    continue
                if int(run_vals.sum()) > thresh_line * mx:
                    filtered[row, s:e] = 1

        # Step 3: C++ 연결 컴포넌트 → bbox
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            filtered, connectivity=8)

        self.bounding_box_U = []
        for lbl in range(1, n_labels):  # 0 = 배경
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            if w * h >= 25:
                self.bounding_box_U.append(Rect(x, y, w, h))

    def extract_3Dbox(self):
        """C++ extract_3Dbox() 완전 동일"""
        new_w = int(self.depth.shape[1] * self.col_scale)
        depth_resize = cv2.resize(self.depth, (new_w, self.depth.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        hist_size = self.depth.shape[0] / self.row_downsample
        bin_width = math.ceil((self.max_dist - self.min_dist) / hist_size)
        num_check = 15

        self.box3Ds = []
        self.bounding_box_D = []

        for b in range(len(self.bounding_box_U)):
            bb_u = self.bounding_box_U[b]
            x = bb_u.x
            width = bb_u.width

            y_up = depth_resize.shape[0]
            y_down = 0
            bin_index_small = bb_u.y
            bin_index_large = bb_u.y + bb_u.height

            depth_in_near = bin_index_small * bin_width + self.min_dist
            depth_of_depth = (bin_index_large - bin_index_small) * bin_width
            depth_in_far = depth_of_depth * 1.3 + depth_in_near

            x_end = min(x + width, depth_resize.shape[1])
            H = depth_resize.shape[0]
            # 해당 열 영역 슬라이스 (C++과 동일하게 모든 행 사용)
            col_slice = depth_resize[:, x:x_end].astype(np.float32) / self.depthScale_ * 1000.0
            in_range = (col_slice >= depth_in_near) & (col_slice <= depth_in_far)
            # num_check 연속 픽셀이 범위 안에 있는 행 찾기 (슬라이딩 합계)
            if in_range.shape[0] >= num_check:
                # np.apply_along_axis + np.convolve 대신 cumsum sliding window (O(n) vs O(n*k))
                arr_int = in_range.astype(np.int32)
                cumsum  = np.cumsum(arr_int, axis=0)
                run     = cumsum[num_check - 1:] - np.vstack([
                    np.zeros((1, arr_int.shape[1]), dtype=np.int32),
                    cumsum[:arr_int.shape[0] - num_check]
                ])
                hit_rows, _ = np.where(run >= num_check)
                if len(hit_rows) > 0:
                    y_up = int(hit_rows.min())
                    y_down = int(hit_rows.max())

            bb_x = x / self.col_scale
            bb_width = width / self.col_scale
            bb_y = y_up
            bb_height = y_down - y_up
            self.bounding_box_D.append(Rect(int(bb_x), int(bb_y),
                                            int(bb_width), int(bb_height)))

            curr_box = Box3D()
            im_frame_x = (x + width / 2) / self.col_scale
            im_frame_x_width = width / self.col_scale

            Y_w = (depth_in_near + depth_in_far) / 2
            im_frame_y = (y_down + y_up) / 2
            im_frame_y_width = y_down - y_up

            self.testx = int(im_frame_x)
            self.testy = int(im_frame_y)
            self.testby = bin_index_small

            curr_box.x = (im_frame_x - self.px) * Y_w / self.fx
            curr_box.y = (im_frame_y - self.py) * Y_w / self.fy
            curr_box.x_width = im_frame_x_width * Y_w / self.fx
            curr_box.y_width = im_frame_y_width * Y_w / self.fy
            curr_box.z = Y_w
            curr_box.z_width = depth_in_far - depth_in_near

            # mm → m 변환
            curr_box.x /= 1000.0
            curr_box.y /= 1000.0
            curr_box.z /= 1000.0
            curr_box.x_width /= 1000.0
            curr_box.y_width /= 1000.0
            curr_box.z_width /= 1000.0

            self.box3Ds.append(curr_box)

    def extract_bird_view(self):
        """C++ extract_bird_view() 동일"""
        hist_size = self.depth.shape[0] // self.row_downsample
        bin_width = math.ceil((self.max_dist - self.min_dist) / float(hist_size))

        self.bounding_box_B = []
        for b in range(len(self.bounding_box_U)):
            bb_u = self.bounding_box_U[b]
            bb_depth = (bb_u.y + bb_u.height) * bin_width / 10.0
            bb_width = bb_depth * bb_u.width / self.fx
            bb_height = bb_u.height * bin_width / 10.0
            bb_x = bb_depth * (bb_u.x / self.col_scale - self.px) / self.fx
            bb_y = bb_depth - 0.5 * bb_height
            self.bounding_box_B.append(
                Rect(int(bb_x), int(bb_y), int(bb_width), int(bb_height))
            )

        self.bird_view = np.zeros((500, 1000, 3), dtype=np.uint8)

    def detect(self):
        """C++ detect() 동일"""
        self.extract_U_map()
        self.extract_bb()
        self.extract_bird_view()

    def track(self):
        """C++ track() 동일"""
        self.tracker.read_bb(self.bounding_box_B, self.bounding_box_D, self.box3Ds)
        self.tracker.check_status(self.box3Ds)
        self.add_tracking_result()

    # ── 시각화 헬퍼 ─────────────────────────

    def display_depth(self):
        """C++ display_depth() 동일 — depth_show 에 저장"""
        if self.depth is None:
            return
        depth_f = self.depth.astype(np.float32)
        _, max_val = cv2.minMaxLoc(depth_f)[:2]
        depth_normalized = cv2.convertScaleAbs(depth_f, alpha=255.0 / max_val if max_val > 0 else 1.0)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_BONE)

        for bb in self.bounding_box_D:
            cv2.rectangle(depth_colored,
                          (bb.x, bb.y),
                          (bb.x + bb.width, bb.y + bb.height),
                          (0, 255, 0), 5)
        self.depth_show = depth_colored

    def display_U_map(self):
        """C++ display_U_map() 동일 — U_map_show 에 저장"""
        if self.U_map is None or not self.show_bounding_box_U:
            return
        u_scaled = np.clip(self.U_map.astype(np.uint16) * 10, 0, 255).astype(np.uint8)
        _, max_val = cv2.minMaxLoc(u_scaled)[:2]
        u_show = cv2.convertScaleAbs(u_scaled, alpha=255.0 / max_val if max_val > 0 else 1.0)
        u_show = cv2.cvtColor(u_show, cv2.COLOR_GRAY2RGB)
        u_show = cv2.applyColorMap(u_show, cv2.COLORMAP_JET)

        for bb in self.bounding_box_U:
            final_bb = Rect(bb.x, bb.y, bb.width, bb.height * 2)
            cv2.rectangle(u_show,
                          (final_bb.x, final_bb.y),
                          (final_bb.x + final_bb.width, final_bb.y + final_bb.height),
                          (0, 255, 0), 1)
        self.U_map_show = u_show

    def display_bird_view(self):
        """C++ display_bird_view() 동일 — bird_view 에 저장"""
        if self.bird_view is None or self.depth is None:
            return
        cx = self.bird_view.shape[1] // 2
        cy = self.bird_view.shape[0]
        center = (cx, cy)

        le = (int(cy * (0 - self.px) / self.fx) + cx, 0)
        re = (int(cy * (self.depth.shape[1] - self.px) / self.fx) + cx, 0)
        cv2.line(self.bird_view, center, le, (0, 255, 0), 3)
        cv2.line(self.bird_view, center, re, (0, 255, 0), 3)

        for b, bb_u in enumerate(self.bounding_box_U):
            if b >= len(self.bounding_box_B):
                break
            final_bb = self.bounding_box_B[b].copy()
            final_bb.y = cy - final_bb.y - final_bb.height
            final_bb.x = final_bb.x + cx
            bb_cx = final_bb.x + final_bb.width // 2
            bb_cy = final_bb.y + final_bb.height // 2
            cv2.rectangle(self.bird_view,
                          (final_bb.x, final_bb.y),
                          (final_bb.x + final_bb.width, final_bb.y + final_bb.height),
                          (0, 0, 255), 3)
            cv2.circle(self.bird_view, (bb_cx, bb_cy), 3, (0, 0, 255), 5)

        self.bird_view = cv2.resize(self.bird_view, None, fx=0.5, fy=0.5)

    def add_tracking_result(self):
        """C++ add_tracking_result() 동일"""
        if self.bird_view is None:
            return
        cx = self.bird_view.shape[1] // 2
        cy = self.bird_view.shape[0]

        for b in range(len(self.tracker.now_bb)):
            if b >= len(self.tracker.now_filter):
                break
            kf = self.tracker.now_filter[b]
            if not kf.is_initialized:
                continue

            est_x = int(kf.output(0) + cx)
            est_y = int(cy - kf.output(1))
            cv2.circle(self.bird_view, (est_x, est_y), 5, (0, 255, 0), 5)

            bb_w = int(kf.output(4))
            bb_h = int(kf.output(5))
            cv2.rectangle(self.bird_view,
                          (est_x - bb_w // 2, est_y - bb_h // 2),
                          (est_x + bb_w // 2, est_y + bb_h // 2),
                          (0, 255, 0), 3)

            vel_x = int(kf.output(2))
            vel_y = int(-kf.output(3))
            cv2.line(self.bird_view, (est_x, est_y),
                     (est_x + vel_x, est_y + vel_y), (255, 255, 255), 3)

            history = self.tracker.now_history[b] if b < len(self.tracker.now_history) else []
            for h in range(1, len(history)):
                sx = int(history[h - 1][0]) + cx
                sy = cy - int(history[h - 1][1])
                ex = int(history[h][0]) + cx
                ey = cy - int(history[h][1])
                cv2.line(self.bird_view, (sx, sy), (ex, ey), (0, 0, 255), 3)
