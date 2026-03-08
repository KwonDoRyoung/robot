"""
FILE: lidar_detector.py
------------------
Python port of lidarDetector.h/.cpp
LiDAR DBSCAN 기반 장애물 감지기
PCL 대신 numpy 사용
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from onboard_detector_python.utils import Box3D
from onboard_detector_python.gpu_dbscan import dbscan_gpu


@dataclass
class Cluster:
    """
    C++ struct Cluster 동일.
    pcl::PointCloud<pcl::PointXYZ> → np.ndarray (Nx3)
    Eigen::Vector4f → np.ndarray shape (4,)
    Eigen::Vector3f → np.ndarray shape (3,)
    Eigen::Matrix3f → np.ndarray shape (3,3)
    """
    cluster_id: int = -1
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    dimensions: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    eigen_vectors: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    eigen_values: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class LidarDetector:
    """
    C++ lidarDetector 클래스 동일.
    pcl::PointCloud → np.ndarray (Nx3)
    DBSCAN 클러스터링 후 3D bbox 생성
    """

    def __init__(self):
        self.eps_: float = 0.5
        self.minPts_: int = 10
        self.cloud_: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.clusters_: List[Cluster] = []
        self.bboxes_: List[Box3D] = []

    def set_params(self, eps: float, min_pts: int):
        """C++ setParams() 동일"""
        self.eps_ = eps
        self.minPts_ = min_pts

    def get_pointcloud(self, cloud: np.ndarray):
        """
        C++ getPointcloud() 동일.
        cloud: np.ndarray shape (N, 3) float32/float64
        """
        self.cloud_ = np.asarray(cloud, dtype=np.float32)

    def lidar_dbscan(self):
        """
        C++ lidarDBSCAN() 완전 동일.
        1. 포인트 → DBSCAN Point 변환
        2. DBSCAN 실행
        3. 클러스터 수집
        4. 각 클러스터의 centroid, dimensions, bbox 계산
           pcl::compute3DCentroid → np.mean
           pcl::getMinMax3D       → np.min/max
        """
        if self.cloud_ is None or len(self.cloud_) == 0:
            return

        pts = np.asarray(self.cloud_, dtype=np.float32)
        import math
        eps = math.sqrt(self.eps_)   # 원본 eps_는 제곱거리 기준
        labels = dbscan_gpu(pts, eps=eps, min_samples=self.minPts_)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # noise(-1) 제거

        clusters_temp = []
        for lbl in unique_labels:
            c = Cluster()
            c.cluster_id = int(lbl) + 1
            c.points = pts[labels == lbl]
            clusters_temp.append(c)

        self.clusters_ = clusters_temp

        # bbox 계산 (C++ 동일)
        bboxes_temp = []
        for cluster in self.clusters_:
            if len(cluster.points) == 0:
                continue

            pts = cluster.points  # Nx3

            # pcl::compute3DCentroid → np.mean (x,y,z,1 형식)
            centroid_xyz = pts.mean(axis=0)  # (3,)
            centroid = np.array([centroid_xyz[0], centroid_xyz[1],
                                 centroid_xyz[2], 1.0], dtype=np.float32)
            cluster.centroid = centroid

            # pcl::getMinMax3D → np.min/max
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            cluster.dimensions = (max_pt - min_pt).astype(np.float32)

            bbox = Box3D()
            bbox.x = float(centroid[0])
            bbox.y = float(centroid[1])
            bbox.z = float(centroid[2])
            bbox.x_width = float(max_pt[0] - min_pt[0])
            bbox.y_width = float(max_pt[1] - min_pt[1])
            bbox.z_width = float(max_pt[2] - min_pt[2])
            bboxes_temp.append(bbox)

        self.bboxes_ = bboxes_temp

    def get_clusters(self) -> List[Cluster]:
        """C++ getClusters() 동일"""
        return self.clusters_

    def get_bboxes(self) -> List[Box3D]:
        """C++ getBBoxes() 동일"""
        return self.bboxes_
