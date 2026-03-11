#!/usr/bin/env python
import rospy
import cv2
import numpy as np
# [Fix 1] TensorRT Python 바인딩이 np.bool(NumPy 1.24에서 삭제)을 사용하므로 패치
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
import torch
import threading
import os
import std_msgs
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from cv_bridge import CvBridge
from ultralytics import YOLO

target_classes = ["person"]

path_curr = os.path.dirname(__file__)
img_topic = "/camera/color/image_raw"
device = "cuda" if torch.cuda.is_available() else "cpu"

# yolo26n.engine (TensorRT) 우선 사용, 없으면 yolo26n.pt, 그것도 없으면 yolo11n.pt
def _find_weight():
    candidates = [
        os.path.join(path_curr, "weights/yolo26n.engine"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            rospy.loginfo(f"[YOLO] Using weight: {c}")
            return c
    raise FileNotFoundError(f"[YOLO] No weight file found in {path_curr}/weights/")

class_names = "config/coco.names"

class yolo_detector:
    def __init__(self):
        print("[onboardDetector]: yolo detector init...")

        self.img_received = False
        self.img_detected = False
        self.detected_img = None
        self.detected_bboxes = []

        # LABEL_NAMES 미리 로드
        self.LABEL_NAMES = []
        names_path = os.path.join(path_curr, class_names)
        if os.path.isfile(names_path):
            with open(names_path, 'r') as f:
                self.LABEL_NAMES = [l.strip() for l in f.readlines()]

        # [Fix 2] Lock — image_callback(subscriber), detect/vis/bbox(timer) 간 race condition 방지
        self._img_lock = threading.Lock()
        self._det_lock = threading.Lock()

        # 모델 로드 (TensorRT engine 또는 pt)
        weight = _find_weight()
        self.use_engine = weight.endswith(".engine")
        # [Fix 3] .engine은 task 자동 감지 불가 → 명시
        self.model = YOLO(weight, task='detect')
        if not self.use_engine:
            self.model.model.eval()

        # CUDA JIT warmup (첫 추론이 수십 초 걸리는 것 방지)
        print("[YOLO] Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, imgsz=640, verbose=False)
        print("[YOLO] Warmup done.")

        self.br = CvBridge()
        # buff_size 크게 설정 → subscriber가 느려도 카메라/rosbag에 backpressure 안걸림
        self.img_sub = rospy.Subscriber(
            img_topic, Image, self.image_callback,
            queue_size=1, buff_size=2**24, tcp_nodelay=True)

        self.img_pub = rospy.Publisher("yolo_detector/detected_image", Image, queue_size=2)
        self.bbox_pub = rospy.Publisher("yolo_detector/detected_bounding_boxes", Detection2DArray, queue_size=2)
        self.time_pub = rospy.Publisher("yolo_detector/yolo_time", std_msgs.msg.Float64, queue_size=1)

        # TRT 없이는 추론이 느리므로 5Hz로 제한 (TRT 있으면 20Hz 가능)
        det_hz = 0.05 if self.use_engine else 0.2
        rospy.Timer(rospy.Duration(det_hz), self.detect_callback)
        rospy.Timer(rospy.Duration(0.1),    self.vis_callback)      # 10Hz
        rospy.Timer(rospy.Duration(det_hz), self.bbox_callback)

    def image_callback(self, msg):
        img = self.br.imgmsg_to_cv2(msg, "bgr8")
        with self._img_lock:
            self.img = img
            self.img_received = True

    def detect_callback(self, event):
        with self._img_lock:
            if not self.img_received:
                return
            img = self.img.copy()
        startTime = rospy.Time.now()
        output = self.inference(img)
        det_img, det_bboxes = self.postprocess(img, output)
        with self._det_lock:
            self.detected_img = det_img
            self.detected_bboxes = det_bboxes
            self.img_detected = True
        self.time_pub.publish((rospy.Time.now() - startTime).to_sec())

        # detect 직후 publish → 별도 타이머 불필요, 동기화 보장
        self.img_pub.publish(self.br.cv2_to_imgmsg(det_img, "bgr8"))
        bboxes_msg = Detection2DArray()
        bboxes_msg.header.stamp = rospy.Time.now()
        for detected_box in det_bboxes:
            if detected_box[4] in target_classes:
                bbox_msg = Detection2D()
                bbox_msg.bbox.center.x = int(detected_box[0])
                bbox_msg.bbox.center.y = int(detected_box[1])
                bbox_msg.bbox.size_x = abs(detected_box[2] - detected_box[0])
                bbox_msg.bbox.size_y = abs(detected_box[3] - detected_box[1])
                bboxes_msg.detections.append(bbox_msg)
        self.bbox_pub.publish(bboxes_msg)

    def vis_callback(self, event):
        # detect_callback에서 publish하므로 유지하되 중복 방지용으로 비워둠
        pass

    def bbox_callback(self, event):
        # detect_callback에서 publish하므로 유지하되 중복 방지용으로 비워둠
        pass

    def inference(self, ori_img):
        # ultralytics가 내부적으로 전처리/추론 수행 (imgsz=640으로 고정)
        preds = self.model(ori_img, imgsz=640, verbose=False)[0]
        return [preds.boxes.xyxyn, preds.boxes.conf, preds.boxes.cls]

    def postprocess(self, ori_img, output):
        H, W, _ = ori_img.shape
        detected_boxes = []
        for i, box in enumerate(output[0]):
            box = box.tolist()
            obj_score = float(output[1][i])
            cls_idx = int(output[2][i])
            category = self.LABEL_NAMES[cls_idx] if cls_idx < len(self.LABEL_NAMES) else str(cls_idx)
            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            detected_boxes.append([x1, y1, x2, y2, category])
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, f"{obj_score:.2f}", (x1, max(y1 - 5, 0)), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(ori_img, category, (x1, max(y1 - 25, 0)), 0, 0.7, (0, 255, 0), 2)
        return ori_img, detected_boxes
