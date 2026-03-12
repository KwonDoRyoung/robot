#!/usr/bin/env python3
"""
dynamic_detector.py
-------------------
Python port of dynamicDetector (LiDAR-Visual dynamic obstacle detection and tracking).
Mirrors dynamicDetector.h/.cpp 1:1.
"""

import math
import threading
import random
from collections import deque
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import rospy
import message_filters

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import ColorRGBA, Header
import cv_bridge

from onboard_detector_python.utils import Box3D, compute_center, compute_std
from onboard_detector_python.kalman_filter import KalmanFilter
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from onboard_detector_python.gpu_dbscan import dbscan_gpu
from onboard_detector_python.uv_detector import UVdetector
from onboard_detector_python.lidar_detector import LidarDetector, Cluster


class DynamicDetector:
    """Python port of onboardDetector::dynamicDetector."""

    NS = "onboard_detector"
    HINT = "[onboardDetector]"

    def __init__(self):
        self._bridge = cv_bridge.CvBridge()
        self._lock = threading.Lock()

        self._init_param()
        self._register_pub()
        self._register_callback()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _gp(self, name, default):
        """Helper: rospy.get_param with default and console echo."""
        full = f"{self.NS}/{name}"
        if rospy.has_param(full):
            val = rospy.get_param(full)
            rospy.loginfo(f"{self.HINT}: {name} = {val}")
        else:
            val = default
            rospy.loginfo(f"{self.HINT}: No {name}. Use default: {default}")
        return val

    def _init_param(self):
        # Localization mode (0=pose, 1=odom)
        self.localization_mode = self._gp("localization_mode", 0)

        # Topic names
        self.depth_topic = self._gp("depth_image_topic", "/camera/depth/image_raw")
        self.color_img_topic = self._gp("color_image_topic", "/camera/color/image_raw")
        self.lidar_topic = self._gp("lidar_pointcloud_topic", "/cloud_registered")
        if self.localization_mode == 0:
            self.pose_topic = self._gp("pose_topic", "/CERLAB/quadcopter/pose")
        else:
            self.odom_topic = self._gp("odom_topic", "/CERLAB/quadcopter/odom")

        # Depth intrinsics (mandatory)
        di = rospy.get_param(f"{self.NS}/depth_intrinsics", None)
        if di is None:
            rospy.logfatal(f"{self.HINT}: Please check camera intrinsics!")
            raise SystemExit(1)
        self.fx, self.fy, self.cx, self.cy = float(di[0]), float(di[1]), float(di[2]), float(di[3])

        # Color intrinsics (mandatory)
        ci = rospy.get_param(f"{self.NS}/color_intrinsics", None)
        if ci is None:
            rospy.logfatal(f"{self.HINT}: Please check camera intrinsics!")
            raise SystemExit(1)
        self.fxC, self.fyC, self.cxC, self.cyC = float(ci[0]), float(ci[1]), float(ci[2]), float(ci[3])

        # Depth params
        self.depth_scale = self._gp("depth_scale_factor", 1000.0)
        self.depth_min = self._gp("depth_min_value", 0.2)
        self.depth_max = self._gp("depth_max_value", 5.0)
        self.raycast_max = self.depth_max
        self.depth_filter_margin = self._gp("depth_filter_margin", 0)
        self.skip_pixel = self._gp("depth_skip_pixel", 1)
        self.img_cols = self._gp("image_cols", 640)
        self.img_rows = self._gp("image_rows", 480)

        # Body-to-sensor transform matrices
        def _load_mat4(name):
            v = rospy.get_param(f"{self.NS}/{name}", None)
            if v is None:
                rospy.logerr(f"{self.HINT}: Please check {name} matrix!")
                return np.eye(4)
            return np.array(v, dtype=np.float64).reshape(4, 4)

        self.body2cam_depth = _load_mat4("body_to_camera_depth")
        self.body2cam_color = _load_mat4("body_to_camera_color")
        self.body2lidar = _load_mat4("body_to_lidar")

        # Time step
        self.dt = self._gp("time_step", 0.033)

        # Ground/roof
        self.ground_height = self._gp("ground_height", 0.1)
        self.roof_height = self._gp("roof_height", 2.0)

        # Voxel filter
        self.voxel_occ_thresh = int(self._gp("voxel_occupied_thresh", 10))

        # DBSCAN visual
        self.db_min_pts = int(self._gp("dbscan_min_points_cluster", 18))
        self.db_epsilon = self._gp("dbscan_search_range_epsilon", 0.3)

        # DBSCAN lidar
        self.lidar_db_min_pts = int(self._gp("lidar_DBSCAN_min_points", 10))
        self.lidar_db_epsilon = self._gp("lidar_DBSCAN_epsilon", 0.2)
        self.down_sample_thresh = int(self._gp("downsample_threshold", 4000))
        self.gaussian_down_rate = int(self._gp("gaussian_downsample_rate", 2))

        # IOU filtering
        self.box_iou_thresh = self._gp("filtering_BBox_IOU_threshold", 0.5)

        # Tracking & association
        self.max_match_range = self._gp("max_match_range", 0.5)
        self.max_match_size_range = self._gp("max_size_diff_range", 0.5)
        fw = rospy.get_param(f"{self.NS}/feature_weight", None)
        if fw is None:
            self.feature_weights = np.array([3.0, 3.0, 0.1, 0.5, 0.5, 0.05, 0.0, 0.0, 0.0])
        else:
            self.feature_weights = np.array(fw, dtype=np.float64)
        self.hist_size = int(self._gp("history_size", 5))
        self.fix_size_hist_thresh = int(self._gp("fix_size_history_threshold", 10))
        self.fix_size_dim_thresh = self._gp("fix_size_dimension_threshold", 0.4)

        # Kalman filter params
        kfp = rospy.get_param(f"{self.NS}/kalman_filter_param", None)
        if kfp is None:
            self.eP = self.eQPos = self.eQVel = self.eQAcc = 0.5
            self.eRPos = self.eRVel = self.eRAcc = 0.5
        else:
            self.eP = kfp[0]; self.eQPos = kfp[1]; self.eQVel = kfp[2]
            self.eQAcc = kfp[3]; self.eRPos = kfp[4]; self.eRVel = kfp[5]; self.eRAcc = kfp[6]
        self.kf_avg_frames = int(self._gp("kalman_filter_averaging_frames", 10))

        # Classification
        self.skip_frame = int(self._gp("frame_skip", 5))
        self.dyna_vel_thresh = self._gp("dynamic_velocity_threshold", 0.35)
        self.dyna_vote_thresh = self._gp("dynamic_voting_threshold", 0.8)
        self.force_dyna_frames = int(self._gp("frames_force_dynamic", 20))
        self.force_dyna_check_range = int(self._gp("frames_force_dynamic_check_range", 30))
        self.dyna_consist_thresh = int(self._gp("dynamic_consistency_threshold", 3))
        if self.hist_size < self.force_dyna_check_range + 1:
            rospy.logerr(f"{self.HINT}: history length is too short to perform force-dynamic")

        # Size constraints
        self.constrain_size = self._gp("target_constrain_size", False)
        tos = rospy.get_param(f"{self.NS}/target_object_size", None)
        self.target_object_sizes = []
        if tos:
            for i in range(0, len(tos), 3):
                self.target_object_sizes.append(np.array([tos[i], tos[i+1], tos[i+2]]))
        mos = rospy.get_param(f"{self.NS}/max_object_size", None)
        self.max_object_size = np.array(mos, dtype=np.float64) if mos else np.array([2.0, 2.0, 2.0])

        # Sensor ranges
        self.local_sensor_range = np.array([5.0, 5.0, 5.0])
        self.local_lidar_range = np.array([0.0, 10.0, 2.0])

        # State variables
        self.depth_image = None               # np.ndarray uint16
        self.position = np.zeros(3)           # robot body position
        self.orientation = np.eye(3)          # robot body rotation matrix
        self.position_depth = np.zeros(3)
        self.orientation_depth = np.eye(3)
        self.position_color = np.zeros(3)
        self.orientation_color = np.eye(3)
        self.position_lidar = np.zeros(3)
        self.orientation_lidar = np.eye(3)
        self.has_sensor_pose = False
        self.latest_cloud_msg = None          # raw sensor_msgs/PointCloud2
        self.lidar_cloud_pts = None           # Nx3 np.ndarray (world frame)
        self.lidar_clusters = []              # List[Cluster]

        # Detection data
        self.uv_bboxes = []
        self.proj_points = []                 # list filled during projection
        self.points_depth = []
        self.proj_points_num = 0
        self.filtered_depth_points = []
        self.db_bboxes = []
        self.pc_clusters_visual = []
        self.pc_cluster_centers_visual = []
        self.pc_cluster_stds_visual = []
        self.filtered_bboxes_before_yolo = []
        self.filtered_bboxes = []
        self.filtered_pc_clusters = []
        self.filtered_pc_cluster_centers = []
        self.filtered_pc_cluster_stds = []
        self.visual_bboxes = []
        self.lidar_bboxes = []
        self.tracked_bboxes = []
        self.dynamic_bboxes = []

        # Tracking
        self.new_detect_flag = False
        self.box_hist = []       # List[deque[Box3D]]
        self.pc_hist = []        # List[deque[List[np.ndarray(3,)]]]
        self.pc_center_hist = [] # List[deque[np.ndarray(3,)]]
        self.filters = []        # List[KalmanFilter]

        # YOLO
        self.yolo_detections = None
        self.detected_color_image = None

        # Sub-module handles (lazy init)
        self._uv_detector = None
        self._lidar_detector = None

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _register_pub(self):
        ns = self.NS

        # Image publishers (plain rospy, no image_transport)
        self._pub_uv_depth = rospy.Publisher(f"{ns}/detected_depth_map", Image, queue_size=10)
        self._pub_u_depth = rospy.Publisher(f"{ns}/detected_u_depth_map", Image, queue_size=10)
        self._pub_uv_bird = rospy.Publisher(f"{ns}/u_depth_bird_view", Image, queue_size=10)
        self._pub_color_img = rospy.Publisher(f"{ns}/detected_color_image", Image, queue_size=10)

        # MarkerArray publishers
        self._pub_uv_bboxes = rospy.Publisher(f"{ns}/uv_bboxes", MarkerArray, queue_size=10)
        self._pub_db_bboxes = rospy.Publisher(f"{ns}/dbscan_bboxes", MarkerArray, queue_size=10)
        self._pub_visual_bboxes = rospy.Publisher(f"{ns}/visual_bboxes", MarkerArray, queue_size=10)
        self._pub_lidar_bboxes = rospy.Publisher(f"{ns}/lidar_bboxes", MarkerArray, queue_size=10)
        self._pub_filtered_before_yolo = rospy.Publisher(f"{ns}/filtered_before_yolo_bboxes", MarkerArray, queue_size=10)
        self._pub_filtered_bboxes = rospy.Publisher(f"{ns}/filtered_bboxes", MarkerArray, queue_size=10)
        self._pub_tracked_bboxes = rospy.Publisher(f"{ns}/tracked_bboxes", MarkerArray, queue_size=10)
        self._pub_dynamic_bboxes = rospy.Publisher(f"{ns}/dynamic_bboxes", MarkerArray, queue_size=10)
        self._pub_history_traj = rospy.Publisher(f"{ns}/history_trajectories", MarkerArray, queue_size=10)
        self._pub_vel_vis = rospy.Publisher(f"{ns}/velocity_visualizaton", MarkerArray, queue_size=10)

        # PointCloud2 publishers
        self._pub_filtered_depth_pts = rospy.Publisher(f"{ns}/filtered_depth_cloud", PointCloud2, queue_size=10)
        self._pub_lidar_clusters = rospy.Publisher(f"{ns}/lidar_clusters", PointCloud2, queue_size=10)
        self._pub_filtered_pts = rospy.Publisher(f"{ns}/filtered_point_cloud", PointCloud2, queue_size=10)
        self._pub_dynamic_pts = rospy.Publisher(f"{ns}/dynamic_point_cloud", PointCloud2, queue_size=10)
        self._pub_raw_dynamic_pts = rospy.Publisher(f"{ns}/raw_dynamic_point_cloud", PointCloud2, queue_size=10)
        self._pub_downsample_pts = rospy.Publisher(f"{ns}/downsampled_point_cloud", PointCloud2, queue_size=10)
        self._pub_raw_lidar_pts = rospy.Publisher(f"{ns}/raw_lidar_point_cloud", PointCloud2, queue_size=10)

    # ------------------------------------------------------------------
    # Subscribers & Timers
    # ------------------------------------------------------------------

    def _register_callback(self):
        # queue_size=1: 최신 프레임만 유지, 구 프레임 즉시 드롭 → 지연 제거
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1)
        lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2, queue_size=1)

        if self.localization_mode == 0:
            pose_sub = message_filters.Subscriber(self.pose_topic, PoseStamped, queue_size=2)
            depth_pose_sync = message_filters.ApproximateTimeSynchronizer(
                [depth_sub, pose_sub], queue_size=2, slop=0.1)
            depth_pose_sync.registerCallback(self._depth_pose_cb)
            lidar_pose_sync = message_filters.ApproximateTimeSynchronizer(
                [lidar_sub, pose_sub], queue_size=2, slop=0.1)
            lidar_pose_sync.registerCallback(self._lidar_pose_cb)
        elif self.localization_mode == 1:
            odom_sub = message_filters.Subscriber(self.odom_topic, Odometry, queue_size=2)
            depth_odom_sync = message_filters.ApproximateTimeSynchronizer(
                [depth_sub, odom_sub], queue_size=2, slop=0.1)
            depth_odom_sync.registerCallback(self._depth_odom_cb)
            lidar_odom_sync = message_filters.ApproximateTimeSynchronizer(
                [lidar_sub, odom_sub], queue_size=2, slop=0.1)
            lidar_odom_sync.registerCallback(self._lidar_odom_cb)
        else:
            rospy.logerr(f"{self.HINT}: Invalid localization mode!")
            raise SystemExit(1)

        # Color image and YOLO
        rospy.Subscriber(self.color_img_topic, Image, self._color_img_cb,
                         queue_size=1, buff_size=2**24)
        rospy.Subscriber("yolo_detector/detected_bounding_boxes", Detection2DArray,
                         self._yolo_detection_cb, queue_size=1)  # 최신 detection만

        # Timers — 타이머 주기를 실제 처리 시간에 맞춤 (30Hz 타이머가 1.4s 콜백 쌓는 문제 제거)
        det_dur   = rospy.Duration(self.dt)
        track_dur = rospy.Duration(self._gp("tracking_time_step", 0.05))
        vis_dur   = rospy.Duration(self._gp("vis_time_step", 0.1))
        rospy.Timer(det_dur,   self._detection_cb)
        rospy.Timer(det_dur,   self._lidar_detection_cb)
        rospy.Timer(track_dur, self._tracking_cb)
        rospy.Timer(track_dur, self._classification_cb)
        rospy.Timer(vis_dur,   self._vis_cb)

        # Service
        from onboard_detector_python.srv import GetDynamicObstacles
        rospy.Service("onboard_detector/get_dynamic_obstacles",
                      GetDynamicObstacles, self._get_dynamic_obstacles_srv)

    # ------------------------------------------------------------------
    # Helper: get camera / lidar pose from body pose message
    # ------------------------------------------------------------------

    def _build_map2body(self, pos, quat_wxyz):
        """Build 4x4 map→body transform from position and quaternion (w,x,y,z)."""
        w, x, y, z = quat_wxyz
        # Rotation from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = pos
        return m

    def _get_camera_pose(self, pos, quat_wxyz):
        """Returns (cam_depth_mat4, cam_color_mat4)."""
        m2b = self._build_map2body(pos, quat_wxyz)
        return m2b @ self.body2cam_depth, m2b @ self.body2cam_color

    def _get_lidar_pose(self, pos, quat_wxyz):
        """Returns lidar_mat4."""
        m2b = self._build_map2body(pos, quat_wxyz)
        return m2b @ self.body2lidar

    def _world_to_body(self, pts_world, pos_body, R_body):
        """World frame 좌표를 base_link frame으로 역변환.
        pos_body: robot position in world (3,)
        R_body: robot rotation matrix body→world (3,3)
        pts_world: Nx3 array in world frame
        Returns: Nx3 array in base_link frame
        """
        return (R_body.T @ (pts_world - pos_body).T).T

    def _box_world_to_body(self, box, pos_body, R_body):
        """Box3D의 x,y,z (world) → base_link 변환. 복사본 반환."""
        b = box.copy()
        p_world = np.array([box.x, box.y, box.z])
        p_body = R_body.T @ (p_world - pos_body)
        b.x, b.y, b.z = float(p_body[0]), float(p_body[1]), float(p_body[2])
        # 속도도 회전만 역변환 (translation 불필요)
        v_world = np.array([box.Vx, box.Vy, box.Vz])
        v_body = R_body.T @ v_world
        b.Vx, b.Vy, b.Vz = float(v_body[0]), float(v_body[1]), float(v_body[2])
        return b

    def _extract_pose(self, pose_msg):
        """Extract (position np.ndarray(3,), quat_wxyz tuple) from PoseStamped."""
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        return np.array([p.x, p.y, p.z]), (o.w, o.x, o.y, o.z)

    def _extract_odom_pose(self, odom_msg):
        """Extract (position, quat_wxyz) from Odometry."""
        p = odom_msg.pose.pose.position
        o = odom_msg.pose.pose.orientation
        return np.array([p.x, p.y, p.z]), (o.w, o.x, o.y, o.z)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _decode_depth_image(self, img_msg):
        """Convert sensor_msgs/Image to uint16 numpy array (mm)."""
        if img_msg.encoding == "32FC1":
            img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")
            img = (img * self.depth_scale).astype(np.uint16)
        else:
            img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="16UC1")
        return img

    def _depth_pose_cb(self, img_msg, pose_msg):
        with self._lock:
            self.depth_image = self._decode_depth_image(img_msg)
            pos, quat = self._extract_pose(pose_msg)
            self.position = pos
            m2b = self._build_map2body(pos, quat)
            self.orientation = m2b[:3, :3]
            cam_depth, cam_color = self._get_camera_pose(pos, quat)
            self.position_depth = cam_depth[:3, 3]
            self.orientation_depth = cam_depth[:3, :3]
            self.position_color = cam_color[:3, 3]
            self.orientation_color = cam_color[:3, :3]

    def _depth_odom_cb(self, img_msg, odom_msg):
        with self._lock:
            self.depth_image = self._decode_depth_image(img_msg)
            pos, quat = self._extract_odom_pose(odom_msg)
            self.position = pos
            m2b = self._build_map2body(pos, quat)
            self.orientation = m2b[:3, :3]
            cam_depth, cam_color = self._get_camera_pose(pos, quat)
            self.position_depth = cam_depth[:3, 3]
            self.orientation_depth = cam_depth[:3, :3]
            self.position_color = cam_color[:3, 3]
            self.orientation_color = cam_color[:3, :3]

    def _process_lidar_cloud(self, cloud_msg, pos, quat):
        """Filter, downsample, transform lidar cloud. Updates self.lidar_cloud_pts."""
        self.latest_cloud_msg = cloud_msg
        self.has_sensor_pose = True

        # Read points from ROS message (ROS Noetic: read_points_numpy 없음 → structured array 변환)
        pts_gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_raw = np.array(list(pts_gen), dtype=np.float32)
        pts = pts_raw.reshape(-1, 3) if pts_raw.size > 0 else pts_raw
        if pts.size == 0:
            self.lidar_cloud_pts = np.zeros((0, 3))
            return

        # ROI filter (base_link frame): x >= 0, 0.3 <= z <= 2.0
        mask = (pts[:, 0] >= 0.0) & (pts[:, 2] >= 0.3) & (pts[:, 2] <= 2.0)
        pts = pts[mask]

        # Gaussian probability downsampling
        sigma = float(self.gaussian_down_rate)
        dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        probs = np.exp(-(dists**2) / (2 * sigma**2))
        rands = np.random.rand(len(pts))
        pts = pts[rands < probs]

        # Transform to world frame (rslidar_points는 이미 base_link 기준 → body2lidar 불필요)
        m2b = self._build_map2body(pos, quat)
        R = m2b[:3, :3]
        t = m2b[:3, 3]
        pts_world = (R @ pts.T).T + t

        # Z filter (ground/roof)
        mask_z = ((pts_world[:, 2] >= self.ground_height) &
                  (pts_world[:, 2] <= self.roof_height))
        pts_world = pts_world[mask_z]

        # Voxel downsampling — Python dict 루프 제거, numpy로 완전 벡터화
        if len(pts_world) > 0:
            leaf = 0.1
            # Adaptive leaf size
            while len(pts_world) > self.down_sample_thresh and leaf <= 2.0:
                leaf *= 1.1
                voxel_idx = np.floor(pts_world / leaf).astype(np.int32)
                # np.unique의 return_index로 첫 번째 포인트만 선택 (dict 루프 완전 대체)
                _, first_idx = np.unique(voxel_idx, axis=0, return_index=True)
                pts_world = pts_world[np.sort(first_idx)]

            # 최종 voxel 대표점
            voxel_idx = np.floor(pts_world / leaf).astype(np.int32)
            _, first_idx = np.unique(voxel_idx, axis=0, return_index=True)
            pts_world = pts_world[np.sort(first_idx)]

        self.lidar_cloud_pts = pts_world

        # Publish downsampled cloud for visualization (world→base_link 역변환)
        pts_body = self._world_to_body(pts_world, t, R) if len(pts_world) > 0 else pts_world
        self._publish_np_pointcloud(pts_body, self._pub_downsample_pts, "base_link")

        # Store lidar sensor pose (base_link = body frame이므로 body pose 그대로 사용)
        self.position_lidar = t.copy()
        self.orientation_lidar = R.copy()

    def _lidar_pose_cb(self, cloud_msg, pose_msg):
        with self._lock:
            pos, quat = self._extract_pose(pose_msg)
            self.position = pos
            m2b = self._build_map2body(pos, quat)
            self.orientation = m2b[:3, :3]
            self._process_lidar_cloud(cloud_msg, pos, quat)

    def _lidar_odom_cb(self, cloud_msg, odom_msg):
        with self._lock:
            pos, quat = self._extract_odom_pose(odom_msg)
            self.position = pos
            m2b = self._build_map2body(pos, quat)
            self.orientation = m2b[:3, :3]
            self._process_lidar_cloud(cloud_msg, pos, quat)

    def _color_img_cb(self, img_msg):
        with self._lock:
            self.detected_color_image = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

    def _yolo_detection_cb(self, det_msg):
        with self._lock:
            self.yolo_detections = det_msg

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _detection_cb(self, event):
        # 입력 데이터 복사 → lock 해제 → 무거운 연산 (numpy/GPU, GIL 해제) → 결과만 lock 하에 반영
        with self._lock:
            if self.depth_image is None or not self.has_sensor_pose:
                return
            depth_img      = self.depth_image.copy()
            pos_depth      = self.position_depth.copy()
            ori_depth      = self.orientation_depth.copy()
            pos_body       = self.position.copy()
            lidar_bboxes   = list(self.lidar_bboxes)
            lidar_clusters = list(self.lidar_clusters)
            yolo           = self.yolo_detections

        # -- lock 없이 무거운 연산 수행 (numpy/GPU가 GIL 해제) --
        # dbscan과 uv_detect는 독립적이므로 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as _ex:
            _db_fut = _ex.submit(self._dbscan_detect_pure,
                                 depth_img, pos_depth, ori_depth, pos_body)
            _uv_fut = _ex.submit(self._uv_detect_pure,
                                 depth_img, pos_depth, ori_depth)
            db_bboxes, pc_vis, pc_centers, pc_stds = _db_fut.result()
            uv_bboxes = _uv_fut.result()
        (filtered_before_yolo, filtered,
         filtered_clusters, filtered_centers, filtered_stds,
         visual_bboxes) = self._filter_lv_bboxes_pure(
            uv_bboxes, db_bboxes, pc_vis, pc_centers, pc_stds,
            lidar_bboxes, lidar_clusters, yolo)

        # -- 결과 반영 (짧은 lock) — 모든 연관 필드를 원자적으로 갱신 --
        with self._lock:
            self.db_bboxes                   = db_bboxes
            self.pc_clusters_visual          = pc_vis
            self.pc_cluster_centers_visual   = pc_centers
            self.pc_cluster_stds_visual      = pc_stds
            self.uv_bboxes                   = uv_bboxes
            self.visual_bboxes               = visual_bboxes
            self.filtered_bboxes_before_yolo = filtered_before_yolo
            self.filtered_bboxes             = filtered
            self.filtered_pc_clusters        = filtered_clusters
            self.filtered_pc_cluster_centers = filtered_centers
            self.filtered_pc_cluster_stds    = filtered_stds
            self.new_detect_flag             = True

    def _lidar_detection_cb(self, event):
        # 입력 데이터 복사 후 lock 해제 → 무거운 연산은 lock 없이 수행
        with self._lock:
            cloud_pts = self.lidar_cloud_pts
            if cloud_pts is None or len(cloud_pts) == 0:
                return
            cloud_pts = cloud_pts.copy()

        if self._lidar_detector is None:
            self._lidar_detector = LidarDetector()
            self._lidar_detector.set_params(self.lidar_db_epsilon, self.lidar_db_min_pts)

        self._lidar_detector.get_pointcloud(cloud_pts)
        self._lidar_detector.lidar_dbscan()
        clusters_raw = self._lidar_detector.get_clusters()
        bboxes_raw = self._lidar_detector.get_bboxes()

        clusters_filtered, bboxes_filtered = [], []
        for i, bbox in enumerate(bboxes_raw):
            if (bbox.x_width > self.max_object_size[0] or
                    bbox.y_width > self.max_object_size[1] or
                    bbox.z_width > self.max_object_size[2]):
                continue
            bboxes_filtered.append(bbox)
            clusters_filtered.append(clusters_raw[i])

        # 결과만 lock 하에 반영
        with self._lock:
            self.lidar_bboxes = bboxes_filtered
            self.lidar_clusters = clusters_filtered

    def _tracking_cb(self, event):
        with self._lock:
            best_match = []
            self._box_association(best_match)
            if best_match:
                self._kalman_filter_and_update_hist(best_match)
            else:
                self.box_hist.clear()
                self.pc_hist.clear()
                self.pc_center_hist.clear()

    def _classification_cb(self, event):
        with self._lock:
            self._classify()
            # dynamic_bboxes를 base_link 기준으로 변환하여 발행
            _dyn = [self._box_world_to_body(b, self.position.copy(), self.orientation.copy())
                    for b in self.dynamic_bboxes]
            self._publish_3d_box(_dyn, self._pub_dynamic_bboxes, 0, 0, 1)

    def _vis_cb(self, event):
        return  # 시각화 비활성화 (detection 성능 확보)
        # lock은 데이터 스냅샷에만 사용 → publish/직렬화 등 무거운 작업은 lock 밖에서
        with self._lock:
            _uv_det        = self._uv_detector
            _color_img     = self.detected_color_image
            _uv_bboxes     = list(self.uv_bboxes)
            _db_bboxes     = list(self.db_bboxes)
            _vis_bboxes    = list(self.visual_bboxes)
            _lidar_bboxes  = list(self.lidar_bboxes)
            _fby_bboxes    = list(self.filtered_bboxes_before_yolo)
            _filt_bboxes   = list(self.filtered_bboxes)
            _track_bboxes  = list(self.tracked_bboxes)
            _dyn_bboxes    = list(self.dynamic_bboxes)
            _lidar_clust   = list(self.lidar_clusters)
            _filt_clust    = list(self.filtered_pc_clusters)
            _fdp           = self.filtered_depth_points
            _fdp_arr       = (_fdp if isinstance(_fdp, np.ndarray) else np.array(_fdp)) if len(_fdp) > 0 else np.zeros((0, 3))
            _dyn_bboxes_snap = list(self.dynamic_bboxes)
            # world→base_link 역변환용 로봇 pose 스냅샷
            _pos_body = self.position.copy()
            _R_body   = self.orientation.copy()

        # -- world→base_link 변환 적용 --
        _uv_bboxes   = [self._box_world_to_body(b, _pos_body, _R_body) for b in _uv_bboxes]
        _db_bboxes   = [self._box_world_to_body(b, _pos_body, _R_body) for b in _db_bboxes]
        _vis_bboxes  = [self._box_world_to_body(b, _pos_body, _R_body) for b in _vis_bboxes]
        _lidar_bboxes= [self._box_world_to_body(b, _pos_body, _R_body) for b in _lidar_bboxes]
        _fby_bboxes  = [self._box_world_to_body(b, _pos_body, _R_body) for b in _fby_bboxes]
        _filt_bboxes = [self._box_world_to_body(b, _pos_body, _R_body) for b in _filt_bboxes]
        _track_bboxes= [self._box_world_to_body(b, _pos_body, _R_body) for b in _track_bboxes]
        _dyn_bboxes  = [self._box_world_to_body(b, _pos_body, _R_body) for b in _dyn_bboxes]
        _dyn_bboxes_snap = _dyn_bboxes  # 이미 변환됨

        # -- lock 밖에서 publish (느린 직렬화/포인트클라우드 변환 포함) --
        self._publish_uv_images_snap(_uv_det)
        self._publish_color_images_snap(_color_img)
        self._publish_3d_box(_uv_bboxes,   self._pub_uv_bboxes,           0, 1, 0)
        self._publish_3d_box(_db_bboxes,   self._pub_db_bboxes,           1, 0, 0)
        self._publish_3d_box(_vis_bboxes,  self._pub_visual_bboxes,       0.3, 0.8, 1.0)
        self._publish_3d_box(_lidar_bboxes,self._pub_lidar_bboxes,        0.5, 0.5, 0.5)
        self._publish_3d_box(_fby_bboxes,  self._pub_filtered_before_yolo,0, 1, 0.5)
        self._publish_3d_box(_filt_bboxes, self._pub_filtered_bboxes,     0, 1, 1)
        self._publish_3d_box(_track_bboxes,self._pub_tracked_bboxes,      1, 1, 0)
        self._publish_3d_box(_dyn_bboxes,  self._pub_dynamic_bboxes,      0, 0, 1)
        self._publish_lidar_clusters_snap(_lidar_clust, _pos_body, _R_body)
        self._publish_filtered_points_snap(_filt_clust, _pos_body, _R_body)
        dynamic_pts = self._get_dynamic_pc_snap(_filt_clust, _dyn_bboxes_snap)
        dyn_arr = np.array(dynamic_pts) if dynamic_pts else np.zeros((0, 3))
        if len(dyn_arr) > 0:
            dyn_arr = self._world_to_body(dyn_arr, _pos_body, _R_body)
        self._publish_np_pointcloud(dyn_arr, self._pub_dynamic_pts, "base_link")
        if len(_fdp_arr) > 0:
            _fdp_arr = self._world_to_body(_fdp_arr, _pos_body, _R_body)
        self._publish_np_pointcloud(_fdp_arr, self._pub_filtered_depth_pts, "base_link")
        self._publish_raw_dynamic_points()
        self._publish_history_traj(_pos_body, _R_body)
        self._publish_vel_vis(_pos_body, _R_body)

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _get_dynamic_obstacles_srv(self, req):
        from onboard_detector_python.srv import GetDynamicObstaclesResponse
        from geometry_msgs.msg import Vector3

        res = GetDynamicObstaclesResponse()
        curr_pos = np.array([req.current_position.x, req.current_position.y, req.current_position.z])

        obstacles = []
        with self._lock:
            for bbox in self.dynamic_bboxes:
                obs_pos = np.array([bbox.x, bbox.y, bbox.z])
                diff = curr_pos - obs_pos
                diff[2] = 0.0
                dist = np.linalg.norm(diff)
                if dist <= req.range:
                    obstacles.append((dist, bbox))

        obstacles.sort(key=lambda x: x[0])
        for dist, bbox in obstacles:
            pos = Vector3(x=bbox.x, y=bbox.y, z=bbox.z)
            vel = Vector3(x=bbox.Vx, y=bbox.Vy, z=0.0)
            size = Vector3(x=bbox.x_width, y=bbox.y_width, z=bbox.z_width)
            res.position.append(pos)
            res.velocity.append(vel)
            res.size.append(size)
        return res

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _uv_detect(self):
        if self._uv_detector is None:
            self._uv_detector = UVdetector()
            self._uv_detector.fx = self.fx
            self._uv_detector.fy = self.fy
            self._uv_detector.px = self.cx
            self._uv_detector.py = self.cy
            self._uv_detector.depthScale_ = self.depth_scale
            self._uv_detector.max_dist = self.raycast_max * 1000

        if self.depth_image is not None:
            self._uv_detector.depth = self.depth_image.copy()
            self._uv_detector.detect()
            self._uv_detector.extract_3Dbox()
            # 구독자 있을 때만 시각화 (display 함수들은 꽤 비쌈)
            vis_needed = (self._pub_uv_depth.get_num_connections() > 0 or
                          self._pub_u_depth.get_num_connections() > 0 or
                          self._pub_uv_bird.get_num_connections() > 0)
            if vis_needed:
                self._uv_detector.display_U_map()
                self._uv_detector.display_bird_view()
                self._uv_detector.display_depth()
            self.uv_bboxes = self._transform_uv_bboxes()

    def _transform_uv_bboxes(self):
        bboxes = []
        for b in self._uv_detector.box3Ds:
            center = np.array([b.x, b.y, b.z])
            size = np.array([b.x_width, b.y_width, b.z_width])
            new_center, new_size = self._transform_bbox(
                center, size, self.position_depth, self.orientation_depth)
            nb = Box3D()
            nb.x, nb.y, nb.z = new_center[0], new_center[1], new_center[2]
            nb.x_width, nb.y_width, nb.z_width = new_size[0], new_size[1], new_size[2]
            bboxes.append(nb)
        return bboxes

    def _dbscan_detect(self):
        self._project_depth_image()
        self._filter_points()
        self.db_bboxes, self.pc_clusters_visual, self.pc_cluster_centers_visual, self.pc_cluster_stds_visual = \
            self._cluster_points_and_bboxes(self.filtered_depth_points)

    def _lidar_detect(self):
        if self._lidar_detector is None:
            self._lidar_detector = LidarDetector()
            self._lidar_detector.set_params(self.lidar_db_epsilon, self.lidar_db_min_pts)

        if self.lidar_cloud_pts is not None and len(self.lidar_cloud_pts) > 0:
            self._lidar_detector.get_pointcloud(self.lidar_cloud_pts)
            self._lidar_detector.lidar_dbscan()
            clusters_raw = self._lidar_detector.get_clusters()
            bboxes_raw = self._lidar_detector.get_bboxes()

            clusters_filtered, bboxes_filtered = [], []
            for i, bbox in enumerate(bboxes_raw):
                if (bbox.x_width > self.max_object_size[0] or
                        bbox.y_width > self.max_object_size[1] or
                        bbox.z_width > self.max_object_size[2]):
                    continue
                bboxes_filtered.append(bbox)
                clusters_filtered.append(clusters_raw[i])
            self.lidar_bboxes = bboxes_filtered
            self.lidar_clusters = clusters_filtered

    # ------------------------------------------------------------------
    # Project depth image to 3D points
    # ------------------------------------------------------------------

    def _project_depth_image(self):
        if self.depth_image is None:
            self.proj_points = []
            self.points_depth = []
            self.proj_points_num = 0
            return

        depth = self.depth_image
        rows, cols = depth.shape
        m = self.depth_filter_margin
        sp = self.skip_pixel
        inv_factor = 1.0 / self.depth_scale
        inv_fx = 1.0 / self.fx
        inv_fy = 1.0 / self.fy

        # Build pixel grid
        vs = np.arange(m, rows - m, sp)
        us = np.arange(m, cols - m, sp)
        vv, uu = np.meshgrid(vs, us, indexing='ij')  # shape (nv, nu)
        # float32 사용: float64 대비 메모리 절반, ARM/CUDA에서 2배 빠름
        raw = depth[vv, uu].astype(np.float32)  # uint16 mm

        depth_vals = raw * inv_factor

        # Handle zero/out-of-range
        invalid_zero = (raw == 0)
        valid = (~invalid_zero) & (depth_vals >= self.depth_min) & (depth_vals <= self.depth_max)

        # Back-project valid points (float32 유지)
        vf = vv[valid].ravel().astype(np.float32)
        uf = uu[valid].ravel().astype(np.float32)
        dv = depth_vals[valid].ravel()

        x_cam = (uf - self.cx) * dv * inv_fx
        y_cam = (vf - self.cy) * dv * inv_fy
        z_cam = dv

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # Nx3, float32
        ori = self.orientation_depth.astype(np.float32)
        pts_world = (ori @ pts_cam.T).T + self.position_depth.astype(np.float32)

        self.proj_points = pts_world          # ndarray (N,3), float32
        self.points_depth = dv                 # ndarray (N,), float32
        self.proj_points_num = len(pts_world)

    # ------------------------------------------------------------------
    # Voxel filter
    # ------------------------------------------------------------------

    def _filter_points(self):
        voxel_filtered = self._voxel_filter()
        if isinstance(voxel_filtered, np.ndarray) and len(voxel_filtered) > 0:
            # numpy 불리언 인덱싱 — Python list comprehension 제거
            mask = (voxel_filtered[:, 2] >= self.ground_height) & (voxel_filtered[:, 2] <= self.roof_height)
            self.filtered_depth_points = voxel_filtered[mask]
        else:
            self.filtered_depth_points = np.zeros((0, 3), dtype=np.float32)

    def _voxel_filter(self):
        if self.proj_points_num == 0:
            return []

        res = 0.1
        pos = self.position
        lr = self.local_sensor_range

        pts = self.proj_points
        depths = self.points_depth
        if not isinstance(pts, np.ndarray):
            pts = np.array(pts)
        if not isinstance(depths, np.ndarray):
            depths = np.array(depths)

        # 범위/지면/raycast 필터 벡터화
        in_range = (
            (pts[:, 0] >= pos[0] - lr[0]) & (pts[:, 0] <= pos[0] + lr[0]) &
            (pts[:, 1] >= pos[1] - lr[1]) & (pts[:, 1] <= pos[1] + lr[1]) &
            (pts[:, 2] >= pos[2] - lr[2]) & (pts[:, 2] <= pos[2] + lr[2]) &
            (pts[:, 2] >= self.ground_height) &
            (depths <= self.raycast_max)
        )
        pts_f = pts[in_range]
        if len(pts_f) == 0:
            return []

        x_voxels = math.ceil(2 * lr[0] / res)
        y_voxels = math.ceil(2 * lr[1] / res)
        z_voxels = math.ceil(2 * lr[2] / res)
        total = x_voxels * y_voxels * z_voxels

        ix = np.floor((pts_f[:, 0] - pos[0] + lr[0]) / res).astype(np.int32)
        iy = np.floor((pts_f[:, 1] - pos[1] + lr[1]) / res).astype(np.int32)
        iz = np.floor((pts_f[:, 2] - pos[2] + lr[2]) / res).astype(np.int32)
        addr = ix * (y_voxels * z_voxels) + iy * z_voxels + iz

        valid = (addr >= 0) & (addr < total)
        pts_f = pts_f[valid]
        addr = addr[valid]

        occ = np.zeros(total, dtype=np.int32)
        np.add.at(occ, addr, 1)

        # 임계값 이상인 voxel의 대표점 1개씩 수집 (numpy array로 반환)
        hit = occ[addr] >= self.voxel_occ_thresh
        _, first_idx = np.unique(addr[hit], return_index=True)
        return pts_f[hit][first_idx]  # ndarray (K,3) — list 변환 비용 제거

    # ------------------------------------------------------------------
    # DBSCAN clustering
    # ------------------------------------------------------------------

    def _cluster_points_and_bboxes(self, points):
        """Returns (bboxes, pc_clusters, pc_cluster_centers, pc_cluster_stds)."""
        if points is None or (hasattr(points, '__len__') and len(points) == 0):
            return [], [], [], []

        # numpy array로 받거나 변환 (list of arrays → np.vstack보다 빠름)
        if isinstance(points, np.ndarray):
            pts_arr = points.astype(np.float32) if points.dtype != np.float32 else points
        else:
            pts_arr = np.array(points, dtype=np.float32)

        # db_epsilon은 원본이 제곱거리 기준이었으므로 sqrt 변환
        eps = math.sqrt(self.db_epsilon)
        labels = dbscan_gpu(pts_arr, eps=eps, min_samples=self.db_min_pts)

        unique_labels = sorted(set(labels) - {-1})
        # numpy 슬라이스 유지 — .tolist() 변환 제거
        clusters_temp = [pts_arr[labels == lbl] for lbl in unique_labels]

        bboxes, pc_clusters, pc_centers, pc_stds = [], [], [], []
        for arr in clusters_temp:
            if len(arr) == 0:
                continue
            xmin, ymin, zmin = arr.min(axis=0)
            xmax, ymax, zmax = arr.max(axis=0)

            box = Box3D()
            box.x = float((xmax + xmin) / 2.0)
            box.y = float((ymax + ymin) / 2.0)
            box.z = float((zmax + zmin) / 2.0)
            box.x_width = float(max(xmax - xmin, 0.1))
            box.y_width = float(max(ymax - ymin, 0.1))
            box.z_width = float(zmax - zmin)

            if (box.x_width > self.max_object_size[0] or
                    box.y_width > self.max_object_size[1] or
                    box.z_width > self.max_object_size[2]):
                continue

            center = arr.mean(axis=0)
            std = np.sqrt(np.mean((arr - center) ** 2, axis=0))
            bboxes.append(box)
            pc_clusters.append(arr)   # numpy array 유지
            pc_centers.append(center)
            pc_stds.append(std)

        return bboxes, pc_clusters, pc_centers, pc_stds

    # ------------------------------------------------------------------
    # Lock-free pure detection methods (데이터를 인자로 받아 self.* 불필요)
    # ------------------------------------------------------------------

    def _dbscan_detect_pure(self, depth_img, pos_depth, ori_depth, pos_body):
        """depth_img 등을 인자로 받아 lock 없이 실행 가능한 버전."""
        # 임시로 self.* 에 복사본 세팅 후 기존 메서드 호출 (thread-local 효과)
        _saved = (self.depth_image, self.position_depth, self.orientation_depth, self.position,
                  self.proj_points, self.points_depth, self.proj_points_num, self.filtered_depth_points)
        self.depth_image      = depth_img
        self.position_depth   = pos_depth
        self.orientation_depth = ori_depth
        self.position         = pos_body

        self._project_depth_image()
        self._filter_points()
        result = self._cluster_points_and_bboxes(self.filtered_depth_points)

        # 복원
        (self.depth_image, self.position_depth, self.orientation_depth, self.position,
         self.proj_points, self.points_depth, self.proj_points_num, self.filtered_depth_points) = _saved
        return result

    def _uv_detect_pure(self, depth_img, pos_depth, ori_depth):
        """UV 탐지 pure 버전."""
        _saved_depth = self.depth_image
        _saved_pd    = self.position_depth
        _saved_od    = self.orientation_depth
        self.depth_image      = depth_img
        self.position_depth   = pos_depth
        self.orientation_depth = ori_depth

        self._uv_detect()
        uv_bboxes = list(self.uv_bboxes)

        self.depth_image      = _saved_depth
        self.position_depth   = _saved_pd
        self.orientation_depth = _saved_od
        return uv_bboxes

    def _filter_lv_bboxes_pure(self, uv_bboxes, db_bboxes, pc_vis, pc_centers, pc_stds,
                                lidar_bboxes, lidar_clusters, yolo):
        """LiDAR-Visual 융합 pure 버전.
        _filter_lv_bboxes()가 self.filtered_* 를 side-effect로 수정하므로,
        입출력에 사용하는 모든 self.* 를 저장했다가 복원한다.
        """
        _saved = (self.uv_bboxes, self.db_bboxes,
                  self.pc_clusters_visual, self.pc_cluster_centers_visual, self.pc_cluster_stds_visual,
                  self.lidar_bboxes, self.lidar_clusters, self.yolo_detections,
                  self.filtered_bboxes, self.filtered_pc_clusters,
                  self.filtered_pc_cluster_centers, self.filtered_pc_cluster_stds,
                  self.filtered_bboxes_before_yolo, self.visual_bboxes)
        self.uv_bboxes                  = uv_bboxes
        self.db_bboxes                  = db_bboxes
        self.pc_clusters_visual         = pc_vis
        self.pc_cluster_centers_visual  = pc_centers
        self.pc_cluster_stds_visual     = pc_stds
        self.lidar_bboxes               = lidar_bboxes
        self.lidar_clusters             = lidar_clusters
        self.yolo_detections            = yolo

        self._filter_lv_bboxes()
        # 결과를 캡처하기 전에 복사본 만들기
        result_before_yolo = list(self.filtered_bboxes_before_yolo)
        result_filtered    = list(self.filtered_bboxes)
        result_clusters    = list(self.filtered_pc_clusters)
        result_centers     = list(self.filtered_pc_cluster_centers)
        result_stds        = list(self.filtered_pc_cluster_stds)
        result_visual      = list(self.visual_bboxes)

        # 모든 self.* 복원 (lock 없이 수정된 필드 포함)
        (self.uv_bboxes, self.db_bboxes,
         self.pc_clusters_visual, self.pc_cluster_centers_visual, self.pc_cluster_stds_visual,
         self.lidar_bboxes, self.lidar_clusters, self.yolo_detections,
         self.filtered_bboxes, self.filtered_pc_clusters,
         self.filtered_pc_cluster_centers, self.filtered_pc_cluster_stds,
         self.filtered_bboxes_before_yolo, self.visual_bboxes) = _saved
        return (result_before_yolo, result_filtered,
                result_clusters, result_centers, result_stds, result_visual)

    # ------------------------------------------------------------------
    # LiDAR-Visual bbox fusion
    # ------------------------------------------------------------------

    def _filter_lv_bboxes(self):
        filtered_bboxes = []
        filtered_clusters = []
        filtered_centers = []
        filtered_stds = []

        visual_bboxes = []
        visual_clusters = []
        visual_centers = []
        visual_stds = []

        lidar_bboxes_temp = []
        lidar_clusters_temp = []
        lidar_centers_temp = []
        lidar_stds_temp = []

        # STEP 1: Fuse UV + DBSCAN (visual)
        for i, uv_bbox in enumerate(self.uv_bboxes):
            best_iou_uv, best_idx_for_uv = 0.0, -1
            best_idx_for_uv = self._get_best_overlap_bbox(uv_bbox, self.db_bboxes)
            if best_idx_for_uv < 0:
                continue
            best_iou_uv = self._cal_box_iou(uv_bbox, self.db_bboxes[best_idx_for_uv])
            if best_iou_uv <= 0:
                continue
            matched_db = self.db_bboxes[best_idx_for_uv]
            best_idx_for_db = self._get_best_overlap_bbox(matched_db, self.uv_bboxes)
            best_iou_db = self._cal_box_iou(matched_db, self.uv_bboxes[best_idx_for_db]) if best_idx_for_db >= 0 else 0.0

            if (best_idx_for_db == i and
                    best_iou_uv > self.box_iou_thresh and
                    best_iou_db > self.box_iou_thresh):
                xmax = max(uv_bbox.x + uv_bbox.x_width/2, matched_db.x + matched_db.x_width/2)
                xmin = min(uv_bbox.x - uv_bbox.x_width/2, matched_db.x - matched_db.x_width/2)
                ymax = max(uv_bbox.y + uv_bbox.y_width/2, matched_db.y + matched_db.y_width/2)
                ymin = min(uv_bbox.y - uv_bbox.y_width/2, matched_db.y - matched_db.y_width/2)
                zmax = max(uv_bbox.z + uv_bbox.z_width/2, matched_db.z + matched_db.z_width/2)
                zmin = min(uv_bbox.z - uv_bbox.z_width/2, matched_db.z - matched_db.z_width/2)
                nb = Box3D()
                nb.x = (xmin + xmax) / 2; nb.y = (ymin + ymax) / 2; nb.z = (zmin + zmax) / 2
                nb.x_width = xmax - xmin; nb.y_width = ymax - ymin; nb.z_width = zmax - zmin
                visual_bboxes.append(nb)
                visual_clusters.append(self.pc_clusters_visual[best_idx_for_uv])
                visual_centers.append(self.pc_cluster_centers_visual[best_idx_for_uv])
                visual_stds.append(self.pc_cluster_stds_visual[best_idx_for_uv])

        self.visual_bboxes = visual_bboxes

        # STEP 2: Prepare lidar bboxes
        for i, lbbox in enumerate(self.lidar_bboxes):
            cluster = self.lidar_clusters[i]
            pts = cluster.points  # Nx3 (numpy array)
            # Python list 변환 제거 — numpy array 그대로 유지
            center = np.array([cluster.centroid[0], cluster.centroid[1], cluster.centroid[2]])
            std = np.sqrt(np.abs(cluster.eigen_values))
            lidar_bboxes_temp.append(lbbox)
            lidar_clusters_temp.append(pts)  # numpy array 유지
            lidar_centers_temp.append(center)
            lidar_stds_temp.append(std)

        # STEP 3: Fuse lidar + visual
        processed_lidar = [False] * len(lidar_bboxes_temp)
        processed_visual = [False] * len(visual_bboxes)

        for i, vbbox in enumerate(visual_bboxes):
            if processed_visual[i]:
                continue
            overlapping_lidar = []
            overlapping_visual_idx = []

            for j, lbbox in enumerate(lidar_bboxes_temp):
                if processed_lidar[j]:
                    continue
                lv_iou = self._cal_box_iou(vbbox, lbbox, ignore_zmin=True)
                if lv_iou > self.box_iou_thresh:
                    overlapping_lidar.append(j)
                    for k in range(len(visual_bboxes)):
                        if processed_visual[k] or k == i:
                            continue
                        iou_k = self._cal_box_iou(visual_bboxes[k], lbbox, ignore_zmin=True)
                        if iou_k > self.box_iou_thresh:
                            overlapping_visual_idx.append(k)

            if not overlapping_lidar:
                filtered_bboxes.append(vbbox)
                filtered_clusters.append(visual_clusters[i])
                filtered_centers.append(visual_centers[i])
                filtered_stds.append(visual_stds[i])
                processed_visual[i] = True
            else:
                xmax = vbbox.x + vbbox.x_width / 2
                xmin = vbbox.x - vbbox.x_width / 2
                ymax = vbbox.y + vbbox.y_width / 2
                ymin = vbbox.y - vbbox.y_width / 2
                zmax = vbbox.z + vbbox.z_width / 2
                zmin = vbbox.z - vbbox.z_width / 2
                fused_pc = list(visual_clusters[i])

                for li in overlapping_lidar:
                    lb = lidar_bboxes_temp[li]
                    xmax = max(xmax, lb.x + lb.x_width / 2)
                    xmin = min(xmin, lb.x - lb.x_width / 2)
                    ymax = max(ymax, lb.y + lb.y_width / 2)
                    ymin = min(ymin, lb.y - lb.y_width / 2)
                    zmax = max(zmax, lb.z + lb.z_width / 2)
                    zmin = min(zmin, lb.z - lb.z_width / 2)
                    fused_pc.extend(lidar_clusters_temp[li])
                    processed_lidar[li] = True

                for vi in overlapping_visual_idx:
                    vb = visual_bboxes[vi]
                    xmax = max(xmax, vb.x + vb.x_width / 2)
                    xmin = min(xmin, vb.x - vb.x_width / 2)
                    ymax = max(ymax, vb.y + vb.y_width / 2)
                    ymin = min(ymin, vb.y - vb.y_width / 2)
                    zmax = max(zmax, vb.z + vb.z_width / 2)
                    zmin = min(zmin, vb.z - vb.z_width / 2)
                    fused_pc.extend(visual_clusters[vi])
                    processed_visual[vi] = True

                fused_center, fused_std = self._calc_pc_feat(fused_pc)
                fb = Box3D()
                fb.x = (xmin + xmax) / 2; fb.y = (ymin + ymax) / 2; fb.z = (zmin + zmax) / 2
                fb.x_width = xmax - xmin; fb.y_width = ymax - ymin; fb.z_width = zmax - zmin
                filtered_bboxes.append(fb)
                filtered_clusters.append(fused_pc)
                filtered_centers.append(fused_center)
                filtered_stds.append(fused_std)
                processed_visual[i] = True

        # STEP 4: Add remaining lidar
        for i, lb in enumerate(lidar_bboxes_temp):
            if processed_lidar[i]:
                continue
            filtered_bboxes.append(lb)
            filtered_clusters.append(lidar_clusters_temp[i])
            filtered_centers.append(lidar_centers_temp[i])
            filtered_stds.append(lidar_stds_temp[i])

        self.filtered_bboxes_before_yolo = list(filtered_bboxes)

        # STEP 5: YOLO integration
        yolo = self.yolo_detections
        if yolo and len(yolo.detections) > 0:
            filtered_bboxes, filtered_clusters, filtered_centers, filtered_stds = \
                self._apply_yolo(filtered_bboxes, filtered_clusters, filtered_centers, filtered_stds, yolo)

        self.filtered_bboxes = filtered_bboxes
        self.filtered_pc_clusters = filtered_clusters
        self.filtered_pc_cluster_centers = filtered_centers
        self.filtered_pc_cluster_stds = filtered_stds

    def _apply_yolo(self, filt_bboxes, filt_clusters, filt_centers, filt_stds, yolo):
        """Integrate YOLO 2D detections into 3D filtered bounding boxes."""
        # Project each filtered 3D bbox onto color image
        proj_2d = []
        for bbox in filt_bboxes:
            center_w = np.array([bbox.x, bbox.y, bbox.z])
            size_w = np.array([bbox.x_width, bbox.y_width, bbox.z_width])
            Rinv = self.orientation_color.T
            tinv = -Rinv @ self.position_color
            center_c, size_c = self._transform_bbox(center_w, size_w, tinv, Rinv)
            tl = center_c - size_c / 2
            br = center_c + size_c / 2
            if tl[2] <= 0:
                proj_2d.append((0, 0, 0, 0))
                continue
            tlX = int((self.fxC * tl[0] + self.cxC * tl[2]) / tl[2])
            tlY = int((self.fyC * tl[1] + self.cyC * tl[2]) / tl[2])
            brX = int((self.fxC * br[0] + self.cxC * br[2]) / br[2])
            brY = int((self.fyC * br[1] + self.cyC * br[2]) / br[2])
            proj_2d.append((tlX, tlY, brX, brY))

        # Draw YOLO boxes on color image
        if self.detected_color_image is not None:
            for det in yolo.detections:
                tlXt = int(det.bbox.center.x)
                tlYt = int(det.bbox.center.y)
                brXt = tlXt + int(det.bbox.size_x)
                brYt = tlYt + int(det.bbox.size_y)
                cv2.rectangle(self.detected_color_image, (tlXt, tlYt), (brXt, brYt), (255, 0, 0), 5)
                cv2.putText(self.detected_color_image, "dynamic", (tlXt, max(tlYt - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Find best 3D box for each YOLO detection
        best_3d_for_yolo = [-1] * len(yolo.detections)
        for i, det in enumerate(yolo.detections):
            tlXt = int(det.bbox.center.x)
            tlYt = int(det.bbox.center.y)
            brXt = tlXt + int(det.bbox.size_x)
            brYt = tlYt + int(det.bbox.size_y)
            best_iou = 0.0
            for j, (tlX, tlY, brX, brY) in enumerate(proj_2d):
                xo = max(0, min(brX, brXt) - max(tlX, tlXt))
                yo = max(0, min(brY, brYt) - max(tlY, tlYt))
                inter = xo * yo
                area1 = (brX - tlX) * (brY - tlY)
                area2 = (brXt - tlXt) * (brYt - tlYt)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_3d_for_yolo[i] = j

        # Build mapping: 3D box → list of yolo indices
        box3d_to_yolo = {}
        for i, idx3d in enumerate(best_3d_for_yolo):
            if 0 <= idx3d < len(filt_bboxes):
                box3d_to_yolo.setdefault(idx3d, []).append(i)

        new_bboxes, new_clusters, new_centers, new_stds = [], [], [], []
        for idx3d in range(len(filt_bboxes)):
            if idx3d not in box3d_to_yolo:
                new_bboxes.append(filt_bboxes[idx3d])
                new_clusters.append(filt_clusters[idx3d])
                new_centers.append(filt_centers[idx3d])
                new_stds.append(filt_stds[idx3d])
                continue

            yolo_indices = box3d_to_yolo[idx3d]
            if len(yolo_indices) == 1:
                filt_bboxes[idx3d].is_dynamic = True
                filt_bboxes[idx3d].is_human = True
                new_bboxes.append(filt_bboxes[idx3d])
                new_clusters.append(filt_clusters[idx3d])
                new_centers.append(filt_centers[idx3d])
                new_stds.append(filt_stds[idx3d])
            else:
                # Multiple YOLO boxes → split the point cloud cluster
                cloud = filt_clusters[idx3d]
                assignment = [-1] * len(cloud)
                for pi, pt_w in enumerate(cloud):
                    pt_c = self.orientation_color.T @ (pt_w - self.position_color)
                    if pt_c[2] <= 0:
                        continue
                    u = int((self.fxC * pt_c[0] + self.cxC * pt_c[2]) / pt_c[2])
                    v = int((self.fyC * pt_c[1] + self.cyC * pt_c[2]) / pt_c[2])
                    best_dist = np.iinfo(np.int32).max
                    for yidx in yolo_indices:
                        det = yolo.detections[yidx]
                        xMin = int(det.bbox.center.x)
                        xMax = xMin + int(det.bbox.size_x)
                        yMin = int(det.bbox.center.y)
                        yMax = yMin + int(det.bbox.size_y)
                        if xMin <= u <= xMax and yMin <= v <= yMax:
                            hdist = max(xMin - u, u - xMax)
                            if hdist < best_dist:
                                best_dist = hdist
                                assignment[pi] = yidx

                used = [False] * len(cloud)
                for yidx in yolo_indices:
                    sub = [cloud[pi] for pi in range(len(cloud))
                           if not used[pi] and assignment[pi] == yidx]
                    for pi in range(len(cloud)):
                        if assignment[pi] == yidx:
                            used[pi] = True
                    if not sub:
                        continue
                    arr = np.array(sub)
                    xmin, ymin, zmin = arr.min(axis=0)
                    xmax, ymax, zmax = arr.max(axis=0)
                    if xmax - xmin <= 0 or ymax - ymin <= 0:
                        continue
                    nb = Box3D()
                    nb.x = (xmin + xmax) / 2; nb.y = (ymin + ymax) / 2; nb.z = (zmin + zmax) / 2
                    nb.x_width = xmax - xmin; nb.y_width = ymax - ymin; nb.z_width = zmax - zmin
                    nb.is_dynamic = True; nb.is_human = True
                    ctr, std = self._calc_pc_feat(sub)
                    new_bboxes.append(nb)
                    new_clusters.append(sub)
                    new_centers.append(ctr)
                    new_stds.append(std)

        return new_bboxes, new_clusters, new_centers, new_stds

    # ------------------------------------------------------------------
    # Data association
    # ------------------------------------------------------------------

    def _box_association(self, best_match):
        num_objs = len(self.filtered_bboxes)
        if not self.box_hist:
            self.box_hist = [deque() for _ in range(num_objs)]
            self.pc_hist = [deque() for _ in range(num_objs)]
            self.pc_center_hist = [deque() for _ in range(num_objs)]
            self.filters = []
            best_match.extend([-1] * num_objs)
            for i in range(num_objs):
                self.box_hist[i].appendleft(deepcopy(self.filtered_bboxes[i]))
                self.pc_hist[i].appendleft(list(self.filtered_pc_clusters[i]))
                self.pc_center_hist[i].appendleft(np.array(self.filtered_pc_cluster_centers[i]))
                states, A, B, H, P, Q, R = self._kalman_filter_matrix_acc(self.filtered_bboxes[i])
                kf = KalmanFilter()
                kf.setup(states, A, B, H, P, Q, R)
                self.filters.append(kf)
        else:
            if self.new_detect_flag:
                self._box_association_helper(best_match)

        self.new_detect_flag = False

    def _box_association_helper(self, best_match):
        num_objs = len(self.filtered_bboxes)
        best_match.extend([0] * num_objs)

        curr_feat = self._gen_feat_helper(self.filtered_bboxes, self.filtered_pc_cluster_centers)

        prev_bboxes, prev_centers = self._get_prev_bboxes()
        prev_feat = self._gen_feat_helper(prev_bboxes, prev_centers)

        prop_bboxes, prop_centers = self._linear_prop()
        prop_feat = self._gen_feat_helper(prop_bboxes, prop_centers)

        self._find_best_match(prev_bboxes, prev_feat, prop_bboxes, prop_feat, curr_feat, best_match)

    def _gen_feat_helper(self, boxes, pc_centers):
        features = []
        fw = self.feature_weights
        for i, box in enumerate(boxes):
            f = np.zeros(10)
            f[0] = (box.x - self.position[0]) * fw[0]
            f[1] = (box.y - self.position[1]) * fw[1]
            f[2] = (box.z - self.position[2]) * fw[2]
            f[3] = box.x_width * fw[3]
            f[4] = box.y_width * fw[4]
            f[5] = box.z_width * fw[5]
            if i < len(pc_centers):
                f[6] = pc_centers[i][0] * fw[6]
                f[7] = pc_centers[i][1] * fw[7]
                f[8] = pc_centers[i][2] * fw[8]
            f = np.where(np.isfinite(f), f, 0.0)
            features.append(f)
        return features

    def _get_prev_bboxes(self):
        prev_boxes = [hist[0] for hist in self.box_hist]
        prev_centers = [hist[0] for hist in self.pc_center_hist]
        return prev_boxes, prev_centers

    def _linear_prop(self):
        prop_boxes = []
        prop_centers = []
        for i, hist in enumerate(self.box_hist):
            pb = deepcopy(hist[0])
            pb.x += pb.Vx * self.dt
            pb.y += pb.Vy * self.dt
            prop_boxes.append(pb)
            pc = np.array(self.pc_center_hist[i][0])
            pc[0] += pb.Vx * self.dt
            pc[1] += pb.Vy * self.dt
            prop_centers.append(pc)
        return prop_boxes, prop_centers

    def _find_best_match(self, prev_bboxes, prev_feat, prop_bboxes, prop_feat, curr_feat, best_match):
        num_objs = len(self.filtered_bboxes)
        for i in range(num_objs):
            best_sim = -1.0
            best_idx = -1
            curr_bbox = self.filtered_bboxes[i]
            for j, prop_bbox in enumerate(prop_bboxes):
                prop_w = max(prop_bbox.x_width, prop_bbox.y_width)
                curr_w = max(curr_bbox.x_width, curr_bbox.y_width)
                if abs(prop_w - curr_w) >= self.max_match_size_range:
                    continue
                dist_xy = math.sqrt((prop_bbox.x - curr_bbox.x)**2 + (prop_bbox.y - curr_bbox.y)**2)
                if dist_xy >= self.max_match_range:
                    continue
                # cosine similarity
                pf_norm = np.linalg.norm(prev_feat[j])
                cf_norm = np.linalg.norm(curr_feat[i])
                ppf_norm = np.linalg.norm(prop_feat[j])
                sim_prev = (prev_feat[j].dot(curr_feat[i]) / (pf_norm * cf_norm)
                            if pf_norm > 0 and cf_norm > 0 else 0.0)
                sim_prop = (prop_feat[j].dot(curr_feat[i]) / (ppf_norm * cf_norm)
                            if ppf_norm > 0 and cf_norm > 0 else 0.0)
                sim = sim_prev + sim_prop
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            best_match[i] = best_idx

    # ------------------------------------------------------------------
    # Kalman filter tracking
    # ------------------------------------------------------------------

    def _kalman_filter_matrix_acc(self, bbox):
        """Build 6-state (x,y,vx,vy,ax,ay) Kalman filter matrices."""
        dt = self.dt
        states = np.array([[bbox.x], [bbox.y], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        A = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1],
        ], dtype=np.float64)
        B = np.zeros((6, 6))
        H = np.eye(6)
        P = np.eye(6) * self.eP
        Q = np.diag([self.eQPos, self.eQPos, self.eQVel, self.eQVel, self.eQAcc, self.eQAcc])
        R = np.diag([self.eRPos, self.eRPos, self.eRVel, self.eRVel, self.eRAcc, self.eRAcc])
        return states, A, B, H, P, Q, R

    def _get_kalman_observation_acc(self, curr_bbox, best_match_idx):
        Z = np.zeros((6, 1))
        Z[0, 0] = curr_bbox.x
        Z[1, 0] = curr_bbox.y
        hist = self.box_hist[best_match_idx]
        k = min(self.kf_avg_frames, len(hist))
        prev = hist[k - 1]
        Z[2, 0] = (curr_bbox.x - prev.x) / (self.dt * k)
        Z[3, 0] = (curr_bbox.y - prev.y) / (self.dt * k)
        Z[4, 0] = (Z[2, 0] - prev.Vx) / (self.dt * k)
        Z[5, 0] = (Z[3, 0] - prev.Vy) / (self.dt * k)
        return Z

    def _kalman_filter_and_update_hist(self, best_match):
        box_hist_temp = []
        pc_hist_temp = []
        pc_center_hist_temp = []
        filters_temp = []
        tracked_temp = []
        num_objs = len(self.filtered_bboxes)
        empty_bh = deque()
        empty_ph = deque()
        empty_pch = deque()

        for i in range(num_objs):
            curr_bbox = self.filtered_bboxes[i]
            if best_match[i] >= 0:
                mi = best_match[i]
                box_hist_temp.append(deque(self.box_hist[mi]))
                pc_hist_temp.append(deque(self.pc_hist[mi]))
                pc_center_hist_temp.append(deque(self.pc_center_hist[mi]))
                filters_temp.append(self.filters[mi])

                Z = self._get_kalman_observation_acc(curr_bbox, mi)
                u = np.zeros((6, 1))
                filters_temp[-1].estimate(Z, u)

                nb = Box3D()
                nb.x = filters_temp[-1].output(0)
                nb.y = filters_temp[-1].output(1)
                nb.z = curr_bbox.z
                nb.Vx = filters_temp[-1].output(2)
                nb.Vy = filters_temp[-1].output(3)
                nb.Ax = filters_temp[-1].output(4)
                nb.Ay = filters_temp[-1].output(5)
                nb.x_width = curr_bbox.x_width
                nb.y_width = curr_bbox.y_width
                nb.z_width = curr_bbox.z_width
                nb.is_dynamic = curr_bbox.is_dynamic
                nb.is_human = curr_bbox.is_human
            else:
                box_hist_temp.append(deque(empty_bh))
                pc_hist_temp.append(deque(empty_ph))
                pc_center_hist_temp.append(deque(empty_pch))
                states, A, B, H, P, Q, R = self._kalman_filter_matrix_acc(curr_bbox)
                kf = KalmanFilter()
                kf.setup(states, A, B, H, P, Q, R)
                filters_temp.append(kf)
                nb = deepcopy(curr_bbox)

            # Trim history
            if len(box_hist_temp[i]) == self.hist_size:
                box_hist_temp[i].pop()
                pc_hist_temp[i].pop()
                pc_center_hist_temp[i].pop()

            box_hist_temp[i].appendleft(nb)
            pc_hist_temp[i].appendleft(list(self.filtered_pc_clusters[i]))
            pc_center_hist_temp[i].appendleft(np.array(self.filtered_pc_cluster_centers[i]))
            tracked_temp.append(nb)

        # Fix size if history is long enough
        for i in range(len(tracked_temp)):
            if len(box_hist_temp[i]) >= self.fix_size_hist_thresh:
                prev = box_hist_temp[i][1]
                curr = tracked_temp[i]
                def _rdiff(a, b): return abs(a - b) / b if b != 0 else 0
                if (_rdiff(curr.x_width, prev.x_width) <= self.fix_size_dim_thresh and
                        _rdiff(curr.y_width, prev.y_width) <= self.fix_size_dim_thresh and
                        _rdiff(curr.z_width, prev.z_width) <= self.fix_size_dim_thresh):
                    tracked_temp[i].x_width = prev.x_width
                    tracked_temp[i].y_width = prev.y_width
                    tracked_temp[i].z_width = prev.z_width
                    box_hist_temp[i][0].x_width = prev.x_width
                    box_hist_temp[i][0].y_width = prev.y_width
                    box_hist_temp[i][0].z_width = prev.z_width

        self.box_hist = box_hist_temp
        self.pc_hist = pc_hist_temp
        self.pc_center_hist = pc_center_hist_temp
        self.filters = filters_temp
        self.tracked_bboxes = tracked_temp

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self):
        dynamic_temp = []

        for i in range(len(self.pc_hist)):
            bh = self.box_hist[i]
            ph = self.pc_hist[i]

            # Case I: YOLO human
            if bh[0].is_human:
                dynamic_temp.append(deepcopy(bh[0]))
                continue

            # Case II: history length check
            cur_frame_gap = min(self.skip_frame, len(ph) - 1)

            # Case III: Force dynamic
            dyna_frames = 0
            if len(bh) > self.force_dyna_check_range:
                for j in range(1, self.force_dyna_check_range + 1):
                    if bh[j].is_dynamic:
                        dyna_frames += 1
            if dyna_frames >= self.force_dyna_frames:
                bh[0].is_dynamic = True
                dynamic_temp.append(deepcopy(bh[0]))
                continue

            curr_pc = ph[0]
            prev_pc = ph[cur_frame_gap]
            Vbox = np.array([
                (bh[0].x - bh[cur_frame_gap].x) / (self.dt * cur_frame_gap) if cur_frame_gap > 0 else 0.0,
                (bh[0].y - bh[cur_frame_gap].y) / (self.dt * cur_frame_gap) if cur_frame_gap > 0 else 0.0,
                (bh[0].z - bh[cur_frame_gap].z) / (self.dt * cur_frame_gap) if cur_frame_gap > 0 else 0.0,
            ])
            Vkf = np.array([bh[0].Vx, bh[0].Vy, 0.0])

            # numpy array로 일괄 변환 (매 포인트마다 np.array() 제거)
            curr_arr = curr_pc if isinstance(curr_pc, np.ndarray) else np.array(curr_pc)
            prev_pts = prev_pc if isinstance(prev_pc, np.ndarray) else (
                np.array(prev_pc) if prev_pc else np.zeros((0, 3))
            )
            num_pts = len(curr_arr)
            votes = 0

            if len(prev_pts) > 0 and num_pts > 0:
                # 벡터화: 각 curr 포인트의 nearest neighbor in prev (N×M 거리 행렬)
                # curr_arr: (N,3), prev_pts: (M,3) → diff: (N,M,3)
                diff = curr_arr[:, None, :] - prev_pts[None, :, :]  # (N,M,3)
                dists2 = (diff ** 2).sum(axis=2)                     # (N,M)
                nn_idx = dists2.argmin(axis=1)                       # (N,)
                nearest_vects = curr_arr - prev_pts[nn_idx]          # (N,3)
                if cur_frame_gap > 0:
                    Vcur_raw = nearest_vects / (self.dt * cur_frame_gap)
                else:
                    Vcur_raw = np.zeros_like(nearest_vects)
                Vcur_xy = np.stack([Vcur_raw[:, 0], Vcur_raw[:, 1], np.zeros(num_pts)], axis=1)  # (N,3)

                vbox_n = np.linalg.norm(Vbox)
                vcur_ns = np.linalg.norm(Vcur_xy, axis=1)           # (N,)
                valid_mask = (vbox_n > 0) & (vcur_ns > 0)
                vel_sim = np.where(
                    valid_mask,
                    (Vcur_xy @ Vbox) / (vcur_ns * vbox_n + 1e-9),
                    0.0
                )
                num_pts -= int((vel_sim < 0).sum())
                votes = int(((vel_sim >= 0) & (vcur_ns > self.dyna_vel_thresh)).sum())

            vote_ratio = float(votes) / float(num_pts) if num_pts > 0 else 0.0
            vel_norm = np.linalg.norm(Vkf)

            if vote_ratio >= self.dyna_vote_thresh and vel_norm >= self.dyna_vel_thresh:
                bh[0].is_dynamic_candidate = True
                dyna_consist = 0
                if len(bh) >= self.dyna_consist_thresh:
                    for j in range(self.dyna_consist_thresh):
                        if bh[j].is_dynamic_candidate or bh[j].is_human or bh[j].is_dynamic:
                            dyna_consist += 1
                if dyna_consist == self.dyna_consist_thresh:
                    bh[0].is_dynamic = True
                    dynamic_temp.append(deepcopy(bh[0]))

        # Size constrain filter
        if self.constrain_size and self.target_object_sizes:
            filtered = []
            for ob in dynamic_temp:
                for ts in self.target_object_sizes:
                    if (abs(ob.x_width - ts[0]) < 0.8 and
                            abs(ob.y_width - ts[1]) < 0.8 and
                            abs(ob.z_width - ts[2]) < 1.0):
                        filtered.append(ob)
                        break
            dynamic_temp = filtered

        self.dynamic_bboxes = dynamic_temp

    # ------------------------------------------------------------------
    # Helper geometry functions
    # ------------------------------------------------------------------

    def _transform_bbox(self, center, size, position, orientation):
        """Transform AABB from camera frame to world frame (or vice-versa)."""
        x, y, z = center
        xw, yw, zw = size[0]/2, size[1]/2, size[2]/2
        corners = np.array([
            [x+xw, y+yw, z+zw], [x+xw, y+yw, z-zw],
            [x+xw, y-yw, z+zw], [x+xw, y-yw, z-zw],
            [x-xw, y+yw, z+zw], [x-xw, y+yw, z-zw],
            [x-xw, y-yw, z+zw], [x-xw, y-yw, z-zw],
        ], dtype=np.float64)
        transformed = (orientation @ corners.T).T + position
        mn = transformed.min(axis=0)
        mx = transformed.max(axis=0)
        new_center = (mn + mx) / 2.0
        new_size = mx - mn
        return new_center, new_size

    def _cal_box_iou(self, box1, box2, ignore_zmin=False):
        b1v = box1.x_width * box1.y_width * box1.z_width
        b2v = box2.x_width * box2.y_width * box2.z_width

        l1Y = (box1.y + box1.y_width/2) - (box2.y - box2.y_width/2)
        l2Y = (box2.y + box2.y_width/2) - (box1.y - box1.y_width/2)
        l1X = (box1.x + box1.x_width/2) - (box2.x - box2.x_width/2)
        l2X = (box2.x + box2.x_width/2) - (box1.x - box1.x_width/2)
        l1Z = (box1.z + box1.z_width/2) - (box2.z - box2.z_width/2)
        l2Z = (box2.z + box2.z_width/2) - (box1.z - box1.z_width/2)

        if ignore_zmin:
            zmin = max(box1.z - box1.z_width/2, box2.z - box2.z_width/2)
            zw1 = box1.z_width/2 + (box1.z - zmin)
            zw2 = box2.z_width/2 + (box2.z - zmin)
            b1v = box1.x_width * box1.y_width * zw1
            b2v = box2.x_width * box2.y_width * zw2
            l1Z = (box1.z + box1.z_width/2) - zmin
            l2Z = (box2.z + box2.z_width/2) - zmin

        ovX = min(l1X, l2X)
        ovY = min(l1Y, l2Y)
        ovZ = min(l1Z, l2Z)

        if max(l1X, l2X) <= max(box1.x_width, box2.x_width):
            ovX = min(box1.x_width, box2.x_width)
        if max(l1Y, l2Y) <= max(box1.y_width, box2.y_width):
            ovY = min(box1.y_width, box2.y_width)
        if max(l1Z, l2Z) <= max(box1.z_width, box2.z_width):
            ovZ = min(box1.z_width, box2.z_width)

        if ovX <= 0 or ovY <= 0 or ovZ <= 0:
            return 0.0

        ov_vol = ovX * ovY * ovZ
        iou = ov_vol / (b1v + b2v - ov_vol)
        return iou

    def _get_best_overlap_bbox(self, curr_bbox, target_bboxes):
        best_iou = 0.0
        best_idx = -1
        for i, tb in enumerate(target_bboxes):
            iou = self._cal_box_iou(curr_bbox, tb)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx

    def _calc_pc_feat(self, pc_cluster):
        """Compute (center, std) of a point cluster. std = population std (÷N)."""
        if not pc_cluster:
            return np.zeros(3), np.zeros(3)
        arr = np.array(pc_cluster)
        center = arr.mean(axis=0)
        variance = np.mean((arr - center)**2, axis=0)
        std = np.sqrt(variance)
        return center, std

    # ------------------------------------------------------------------
    # Dynamic point cloud extraction
    # ------------------------------------------------------------------

    def _get_dynamic_pc(self):
        dynamic_pts = []
        for cluster in self.filtered_pc_clusters:
            for pt in cluster:
                p = np.array(pt)
                for db in self.dynamic_bboxes:
                    if (abs(p[0] - db.x) <= db.x_width/2 and
                            abs(p[1] - db.y) <= db.y_width/2 and
                            abs(p[2] - db.z) <= db.z_width/2):
                        dynamic_pts.append(p)
                        break
        return dynamic_pts

    # ------------------------------------------------------------------
    # Visualization publishers
    # ------------------------------------------------------------------

    def _publish_uv_images(self):
        if self._uv_detector is None:
            return
        try:
            now = rospy.Time.now()
            if hasattr(self._uv_detector, 'depth_show') and self._uv_detector.depth_show is not None:
                msg = self._bridge.cv2_to_imgmsg(self._uv_detector.depth_show, encoding="bgr8")
                msg.header.stamp = now
                self._pub_uv_depth.publish(msg)
            if hasattr(self._uv_detector, 'U_map_show') and self._uv_detector.U_map_show is not None:
                msg = self._bridge.cv2_to_imgmsg(self._uv_detector.U_map_show, encoding="bgr8")
                msg.header.stamp = now
                self._pub_u_depth.publish(msg)
            if hasattr(self._uv_detector, 'bird_view') and self._uv_detector.bird_view is not None:
                msg = self._bridge.cv2_to_imgmsg(self._uv_detector.bird_view, encoding="bgr8")
                msg.header.stamp = now
                self._pub_uv_bird.publish(msg)
        except Exception:
            pass

    def _publish_color_images(self):
        if self.detected_color_image is None:
            return
        try:
            msg = self._bridge.cv2_to_imgmsg(self.detected_color_image, encoding="rgb8")
            msg.header.stamp = rospy.Time.now()
            self._pub_color_img.publish(msg)
        except Exception:
            pass

    def _make_point_msg(self, x, y, z):
        p = Point()
        p.x = x; p.y = y; p.z = z
        return p

    def _publish_3d_box(self, boxes, publisher, r, g, b):
        markers = MarkerArray()
        now = rospy.Time.now()
        for i, box in enumerate(boxes):
            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = now
            line.ns = "box3D"
            line.id = i
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.06
            line.color.r = r; line.color.g = g; line.color.b = b; line.color.a = 1.0
            line.lifetime = rospy.Duration(0.06)
            line.pose.orientation.w = 1.0
            line.pose.position.x = box.x
            line.pose.position.y = box.y
            top = box.z + box.z_width / 2.0
            z_off = top / 2.0
            line.pose.position.z = z_off

            xw = box.x_width / 2.0
            yw = box.y_width / 2.0
            corners = [
                (-xw, -yw, -z_off), (-xw,  yw, -z_off), ( xw,  yw, -z_off), ( xw, -yw, -z_off),
                (-xw, -yw,  z_off), (-xw,  yw,  z_off), ( xw,  yw,  z_off), ( xw, -yw,  z_off),
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b_idx in edges:
                line.points.append(self._make_point_msg(*corners[a]))
                line.points.append(self._make_point_msg(*corners[b_idx]))
            markers.markers.append(line)
        publisher.publish(markers)

    def _publish_np_pointcloud(self, pts, publisher, frame_id):
        """Publish Nx3 float32 numpy array as PointCloud2."""
        if pts is None or len(pts) == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        header = Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.now()
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
        ]
        cloud_msg = pc2.create_cloud(header, fields, pts.tolist())
        publisher.publish(cloud_msg)

    def _publish_history_traj(self, pos_body=None, R_body=None):
        traj_msg = MarkerArray()
        count = 0
        for i, bh in enumerate(self.box_hist):
            if len(bh) <= 1:
                continue
            traj = Marker()
            traj.header.frame_id = "base_link"
            traj.header.stamp = rospy.Time.now()
            traj.ns = "dynamic_detector"
            traj.id = count
            traj.type = Marker.LINE_LIST
            traj.scale.x = 0.03
            traj.color.a = 1.0; traj.color.r = 0.0; traj.color.g = 1.0; traj.color.b = 0.0
            traj.pose.orientation.w = 1.0
            bh_list = list(bh)
            for j in range(len(bh_list) - 1):
                b1, b2 = bh_list[j], bh_list[j+1]
                if pos_body is not None and R_body is not None:
                    p1 = R_body.T @ (np.array([b1.x, b1.y, b1.z]) - pos_body)
                    p2 = R_body.T @ (np.array([b2.x, b2.y, b2.z]) - pos_body)
                    traj.points.append(self._make_point_msg(p1[0], p1[1], p1[2]))
                    traj.points.append(self._make_point_msg(p2[0], p2[1], p2[2]))
                else:
                    traj.points.append(self._make_point_msg(b1.x, b1.y, b1.z))
                    traj.points.append(self._make_point_msg(b2.x, b2.y, b2.z))
            traj_msg.markers.append(traj)
            count += 1
        self._pub_history_traj.publish(traj_msg)

    def _publish_vel_vis(self, pos_body=None, R_body=None):
        vel_msg = MarkerArray()
        for i, tb in enumerate(self.tracked_bboxes):
            vm = Marker()
            vm.header.frame_id = "base_link"
            vm.header.stamp = rospy.Time.now()
            vm.ns = "dynamic_detector"
            vm.id = i
            vm.type = Marker.TEXT_VIEW_FACING
            if pos_body is not None and R_body is not None:
                p = R_body.T @ (np.array([tb.x, tb.y, tb.z]) - pos_body)
                vm.pose.position.x = p[0]
                vm.pose.position.y = p[1]
                vm.pose.position.z = p[2] + tb.z_width / 2.0 + 0.3
            else:
                vm.pose.position.x = tb.x
                vm.pose.position.y = tb.y
                vm.pose.position.z = tb.z + tb.z_width / 2.0 + 0.3
            vm.scale.x = vm.scale.y = vm.scale.z = 0.15
            vm.color.a = 1.0; vm.color.r = 1.0
            vm.lifetime = rospy.Duration(0.1)
            vn = math.sqrt(tb.Vx**2 + tb.Vy**2)
            vm.text = f"Vx={tb.Vx:.2f}, Vy={tb.Vy:.2f}, |V|={vn:.2f}"
            vel_msg.markers.append(vm)
        self._pub_vel_vis.publish(vel_msg)

    # -- snapshot 버전: lock 밖에서 호출, 인수로 데이터를 받음 --
    def _publish_uv_images_snap(self, uv_det):
        if uv_det is None:
            return
        try:
            now = rospy.Time.now()
            if hasattr(uv_det, 'depth_show') and uv_det.depth_show is not None:
                msg = self._bridge.cv2_to_imgmsg(uv_det.depth_show, encoding="bgr8")
                msg.header.stamp = now
                self._pub_uv_depth.publish(msg)
            if hasattr(uv_det, 'U_map_show') and uv_det.U_map_show is not None:
                msg = self._bridge.cv2_to_imgmsg(uv_det.U_map_show, encoding="bgr8")
                msg.header.stamp = now
                self._pub_u_depth.publish(msg)
            if hasattr(uv_det, 'bird_view') and uv_det.bird_view is not None:
                msg = self._bridge.cv2_to_imgmsg(uv_det.bird_view, encoding="bgr8")
                msg.header.stamp = now
                self._pub_uv_bird.publish(msg)
        except Exception:
            pass

    def _publish_color_images_snap(self, color_img):
        if color_img is None:
            return
        try:
            msg = self._bridge.cv2_to_imgmsg(color_img, encoding="rgb8")
            msg.header.stamp = rospy.Time.now()
            self._pub_color_img.publish(msg)
        except Exception:
            pass

    def _publish_lidar_clusters_snap(self, lidar_clusters, pos_body=None, R_body=None):
        pts_xyzrgb = []
        for cluster in lidar_clusters:
            random.seed(cluster.cluster_id)
            rc = random.random(); gc = random.random(); bc = random.random()
            for pt in cluster.points:
                if pos_body is not None and R_body is not None:
                    p = R_body.T @ (np.array(pt[:3]) - pos_body)
                    pts_xyzrgb.append((p[0], p[1], p[2], rc, gc, bc))
                else:
                    pts_xyzrgb.append((pt[0], pt[1], pt[2], rc, gc, bc))
        if not pts_xyzrgb:
            return
        header = Header(frame_id="base_link", stamp=rospy.Time.now())
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
            pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
            pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1),
        ]
        self._pub_lidar_clusters.publish(pc2.create_cloud(header, fields, pts_xyzrgb))

    def _publish_filtered_points_snap(self, filtered_pc_clusters, pos_body=None, R_body=None):
        pts_xyzrgb = []
        for cluster in filtered_pc_clusters:
            for pt in cluster:
                if pos_body is not None and R_body is not None:
                    p = R_body.T @ (np.array([float(pt[0]), float(pt[1]), float(pt[2])]) - pos_body)
                    pts_xyzrgb.append((p[0], p[1], p[2], 0.5, 0.5, 0.5))
                else:
                    pts_xyzrgb.append((float(pt[0]), float(pt[1]), float(pt[2]), 0.5, 0.5, 0.5))
        if not pts_xyzrgb:
            return
        header = Header(frame_id="base_link", stamp=rospy.Time.now())
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
            pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
            pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1),
        ]
        self._pub_filtered_pts.publish(pc2.create_cloud(header, fields, pts_xyzrgb))

    def _get_dynamic_pc_snap(self, filtered_pc_clusters, dynamic_bboxes):
        dynamic_pts = []
        for cluster in filtered_pc_clusters:
            for pt in cluster:
                p = np.array(pt)
                for db in dynamic_bboxes:
                    if (abs(p[0] - db.x) <= db.x_width/2 and
                            abs(p[1] - db.y) <= db.y_width/2 and
                            abs(p[2] - db.z) <= db.z_width/2):
                        dynamic_pts.append(p)
                        break
        return dynamic_pts

    # -- 기존 함수 (lock 안에서 호출되던 버전, 다른 곳 참조용으로 유지) --
    def _publish_lidar_clusters(self):
        pts_xyzrgb = []
        for cluster in self.lidar_clusters:
            random.seed(cluster.cluster_id)
            rc = random.random(); gc = random.random(); bc = random.random()
            for pt in cluster.points:
                pts_xyzrgb.append((pt[0], pt[1], pt[2], rc, gc, bc))
        if not pts_xyzrgb:
            return
        header = Header(frame_id="base_link", stamp=rospy.Time.now())
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
            pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
            pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1),
        ]
        msg = pc2.create_cloud(header, fields, pts_xyzrgb)
        self._pub_lidar_clusters.publish(msg)

    def _publish_filtered_points(self):
        pts_xyzrgb = []
        for cluster in self.filtered_pc_clusters:
            for pt in cluster:
                pts_xyzrgb.append((float(pt[0]), float(pt[1]), float(pt[2]), 0.5, 0.5, 0.5))
        if not pts_xyzrgb:
            return
        header = Header(frame_id="base_link", stamp=rospy.Time.now())
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
            pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
            pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1),
        ]
        msg = pc2.create_cloud(header, fields, pts_xyzrgb)
        self._pub_filtered_pts.publish(msg)

    def _publish_raw_dynamic_points(self):
        if self.latest_cloud_msg is None:
            return
        try:
            pts_gen = pc2.read_points(self.latest_cloud_msg,
                                      field_names=("x", "y", "z"), skip_nans=True)
            pts = np.array(list(pts_gen), dtype=np.float32)

            if self.has_sensor_pose and len(pts) > 0:
                R = self.orientation_lidar
                t = self.position_lidar
                global_pts = (R @ pts.T).T + t
                # world→base_link 역변환 후 publish
                body_pts = self._world_to_body(global_pts, t, R)
                self._publish_np_pointcloud(body_pts, self._pub_raw_lidar_pts, "base_link")
            else:
                global_pts = pts

            dyn_pts = []
            for box in self.dynamic_bboxes:
                if not box.is_dynamic:
                    continue
                xmin, xmax = box.x - box.x_width/2, box.x + box.x_width/2
                ymin, ymax = box.y - box.y_width/2, box.y + box.y_width/2
                zmin, zmax = box.z - box.z_width/2, box.z + box.z_width/2
                mask = ((global_pts[:, 0] >= xmin) & (global_pts[:, 0] <= xmax) &
                        (global_pts[:, 1] >= ymin) & (global_pts[:, 1] <= ymax) &
                        (global_pts[:, 2] >= zmin) & (global_pts[:, 2] <= zmax))
                dyn_pts.extend(global_pts[mask].tolist())

            if dyn_pts:
                dyn_arr = np.array(dyn_pts)
                dyn_arr = self._world_to_body(dyn_arr, t, R)
                self._publish_np_pointcloud(dyn_arr, self._pub_raw_dynamic_pts, "base_link")
        except Exception as e:
            rospy.logerr(f"publishRawDynamicPoints error: {e}")

    # ------------------------------------------------------------------
    # Public API (mirrors C++ user functions)
    # ------------------------------------------------------------------

    def get_dynamic_obstacles(self, robot_size=None):
        """Return list of Box3D expanded by robot_size."""
        if robot_size is None:
            robot_size = np.zeros(3)
        result = []
        with self._lock:
            for box in self.dynamic_bboxes:
                b = deepcopy(box)
                b.x_width += robot_size[0]
                b.y_width += robot_size[1]
                b.z_width += robot_size[2]
                result.append(b)
        return result

    def get_dynamic_obstacles_hist(self, robot_size=None):
        """Return (pos_hist, vel_hist, size_hist) for dynamic obstacles."""
        if robot_size is None:
            robot_size = np.zeros(3)
        pos_hist, vel_hist, size_hist = [], [], []
        with self._lock:
            for bh in self.box_hist:
                if not (bh[0].is_dynamic or bh[0].is_human):
                    continue
                if self.constrain_size and self.target_object_sizes:
                    found = False
                    for ts in self.target_object_sizes:
                        if (abs(bh[0].x_width - ts[0]) < 0.8 and
                                abs(bh[0].y_width - ts[1]) < 0.8 and
                                abs(bh[0].z_width - ts[2]) < 1.0):
                            found = True; break
                    if not found:
                        continue
                op, ov, os_ = [], [], []
                for b in bh:
                    op.append(np.array([b.x, b.y, b.z]))
                    ov.append(np.array([b.Vx, b.Vy, 0.0]))
                    os_.append(np.array([b.x_width + robot_size[0],
                                         b.y_width + robot_size[1],
                                         b.z_width + robot_size[2]]))
                pos_hist.append(op); vel_hist.append(ov); size_hist.append(os_)
        return pos_hist, vel_hist, size_hist
