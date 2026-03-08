"""
gpu_dbscan.py
-------------
Jetson Orin GPU(CUDA)를 활용한 DBSCAN 구현.
torch를 이용해 거리 행렬을 GPU에서 계산하고, BFS로 클러스터를 묶는다.

포인트 수가 적을 때(< 200)는 sklearn CPU가 빠를 수 있으므로 임계값으로 분기.
"""

import numpy as np
import torch

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GPU_THRESHOLD = 150  # 이 이하는 CPU sklearn 사용


def dbscan_gpu(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    GPU 기반 DBSCAN.
    pts: (N, 3) float32 ndarray
    반환: (N,) int32 라벨 배열 (-1 = noise, 0,1,2,... = 클러스터 ID)
    """
    n = len(pts)
    if n == 0:
        return np.array([], dtype=np.int32)

    if n < _GPU_THRESHOLD:
        # 소규모는 sklearn이 더 빠름
        from sklearn.cluster import DBSCAN as SkDBSCAN
        return SkDBSCAN(eps=eps, min_samples=min_samples,
                        algorithm='ball_tree', n_jobs=1).fit_predict(pts).astype(np.int32)

    t = torch.from_numpy(pts.astype(np.float32)).to(_DEVICE)  # (N,3)

    # 배치별 거리 행렬 계산 (메모리 절약)
    batch = 512
    neighbor_mask = torch.zeros((n, n), dtype=torch.bool, device=_DEVICE)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        diff = t[start:end].unsqueeze(1) - t.unsqueeze(0)   # (B, N, 3)
        dist2 = (diff * diff).sum(dim=2)                     # (B, N)
        neighbor_mask[start:end] = dist2 <= (eps * eps)

    # 이웃 수 계산 → core points
    neighbor_count = neighbor_mask.sum(dim=1)                 # (N,)
    is_core = neighbor_count >= min_samples                   # (N,) bool

    # CPU로 가져와 BFS 클러스터링
    neighbor_mask_cpu = neighbor_mask.cpu().numpy()
    is_core_cpu = is_core.cpu().numpy()

    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1 or not is_core_cpu[i]:
            continue
        # BFS
        queue = [i]
        labels[i] = cluster_id
        while queue:
            cur = queue.pop()
            neighbors = np.where(neighbor_mask_cpu[cur])[0]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if is_core_cpu[nb]:
                        queue.append(nb)
        cluster_id += 1

    return labels
