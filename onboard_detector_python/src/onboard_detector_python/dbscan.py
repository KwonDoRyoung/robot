"""
FILE: dbscan.py
------------------
Python port of dbscan.h/.cpp
C++ 코드와 완전히 동일한 로직 (squared distance 비교 포함)

중요: calculateDistance()는 제곱 거리를 반환하며,
m_epsilon과 직접 비교한다 (sqrt 없음 — C++ 원본 동일).
따라서 epsilon=0.05 는 실제 거리 약 0.224m에 해당한다.
"""

UNCLASSIFIED = -1
CORE_POINT = 1
BORDER_POINT = 2
NOISE = -2
SUCCESS = 0
FAILURE = -3


class Point:
    """C++ Point_ struct 동일"""
    __slots__ = ('x', 'y', 'z', 'clusterID')

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.clusterID = UNCLASSIFIED


class DBSCAN:
    """
    C++ DBSCAN 클래스의 Python 완전 동일 포팅.
    m_points 는 Point 객체 리스트이며, run() 실행 후 clusterID 가 인플레이스로 설정된다.
    """

    def __init__(self, min_pts: int, eps: float, points: list):
        """
        min_pts: 클러스터를 형성하기 위한 최소 포인트 수
        eps: 제곱 거리 임계값 (C++ m_epsilon 동일 — sqrt 없이 직접 비교)
        points: Point 객체 리스트
        """
        self.m_minPoints = min_pts
        self.m_epsilon = eps       # 제곱 거리와 비교됨
        self.m_points = points
        self.m_pointSize = len(points)

    def run(self) -> int:
        """C++ run() 동일 — 모든 점에 대해 클러스터 확장 시도"""
        clusterID = 1
        for point in self.m_points:
            if point.clusterID == UNCLASSIFIED:
                if self.expandCluster(point, clusterID) != FAILURE:
                    clusterID += 1
        return 0

    def expandCluster(self, point: Point, clusterID: int) -> int:
        """C++ expandCluster() 완전 동일"""
        clusterSeeds = self.calculateCluster(point)

        if len(clusterSeeds) < self.m_minPoints:
            point.clusterID = NOISE
            return FAILURE
        else:
            index = 0
            indexCorePoint = 0
            for idx in clusterSeeds:
                self.m_points[idx].clusterID = clusterID
                if (self.m_points[idx].x == point.x and
                        self.m_points[idx].y == point.y and
                        self.m_points[idx].z == point.z):
                    indexCorePoint = index
                index += 1

            clusterSeeds.pop(indexCorePoint)

            i = 0
            n = len(clusterSeeds)
            while i < n:
                clusterNeighbors = self.calculateCluster(self.m_points[clusterSeeds[i]])

                if len(clusterNeighbors) >= self.m_minPoints:
                    for neighbor_idx in clusterNeighbors:
                        if (self.m_points[neighbor_idx].clusterID == UNCLASSIFIED or
                                self.m_points[neighbor_idx].clusterID == NOISE):
                            if self.m_points[neighbor_idx].clusterID == UNCLASSIFIED:
                                clusterSeeds.append(neighbor_idx)
                                n = len(clusterSeeds)
                            self.m_points[neighbor_idx].clusterID = clusterID
                i += 1

            return SUCCESS

    def calculateCluster(self, point: Point) -> list:
        """C++ calculateCluster() 동일 — 제곱 거리 기준으로 이웃 인덱스 반환"""
        clusterIndex = []
        for index, p in enumerate(self.m_points):
            if self.calculateDistance(point, p) <= self.m_epsilon:
                clusterIndex.append(index)
        return clusterIndex

    def calculateDistance(self, pointCore: Point, pointTarget: Point) -> float:
        """
        C++ calculateDistance() 완전 동일.
        제곱 유클리드 거리 반환 (sqrt 없음).
        pow(dx,2) + pow(dy,2) + pow(dz,2)
        """
        dx = pointCore.x - pointTarget.x
        dy = pointCore.y - pointTarget.y
        dz = pointCore.z - pointTarget.z
        return dx * dx + dy * dy + dz * dz

    def get_total_point_size(self) -> int:
        return self.m_pointSize

    def get_minimum_cluster_size(self) -> int:
        return self.m_minPoints

    def get_epsilon_size(self) -> float:
        return self.m_epsilon
