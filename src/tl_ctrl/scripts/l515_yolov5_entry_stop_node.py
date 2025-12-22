#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv5 Obstacle Stop Node (RGB only)
- 입력: RGB 이미지 토픽 -> YOLOv5로 person/car 감지
- ROI(프레임 하단 + 중앙 폭) 내에서 충분히 큰 박스가 발견되면 침입자로 간주
- 프레임 투표 + 유지/쿨다운 로직으로 /emergency_stop (Bool) 발행
- (옵션) 즉시 0속도 Twist 1회 발행

[필수]
- yolov5 레포 + 가중치(.pt), OpenCV, PyTorch, ROS, cv_bridge

[주요 파라미터(rosparam)]
~image_topic         (str)  입력 컬러 이미지 토픽 (default: /camera/color/image_raw)
~emergency_topic     (str)  출력 Bool 토픽 (default: /emergency_stop)
~publish_cmd_vel     (bool) True면 정지 트위스트 1회 발행 (default: False)
~cmd_vel_topic       (str)  /cmd_vel (default)

~yolov5_dir          (str)  yolov5 루트 경로
~weights             (str)  가중치 파일 경로(절대/상대 모두 허용)
~imgsz               (int)  640
~conf_thres          (float)0.40
~iou_thres           (float)0.45
~max_det             (int)  50

~stop_classes        (list[str]) YAML 리스트(예: ['person','car']) 또는
~stop_classes_csv    (str)  콤마구분 문자열(예: "person,car")

~roi_bottom_ratio    (float)하단 비율(0~1). 0.35면 프레임 하단 35%만 사용
~roi_center_ratio    (float)중앙 폭 비율(0~1). 0.6이면 중앙 60%만 사용
~min_box_area_ratio  (float)박스 면적/프레임 면적의 최소 비율(예: 0.004=0.4%)
~min_conf_override   (float)클래스별 기본 conf를 덮어쓸 최소 conf(옵션)

~vote_window         (int)  투표창 프레임 수 (예: 5)
~vote_threshold      (int)  창 내 True 합계 임계 (예: 3)
~stop_hold_sec       (float)정지 유지 시간
~cooldown_sec        (float)해제 후 재트리거 쿨다운

~show_debug          (bool) 디버그 창
~window_name         (str)  창 이름
"""

import os
import sys
import time as ptime
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


def _logi(m): rospy.loginfo(m)
def _logw(m): rospy.logwarn(m)
def _loge(m): rospy.logerr(m)


def _parse_stop_classes():
    # 우선 리스트 파라미터 시도
    cls_list = rospy.get_param("~stop_classes", None)
    if isinstance(cls_list, list) and all(isinstance(x, str) for x in cls_list):
        return [c.strip().lower() for c in cls_list]
    # 콤마구분 문자열 fallback
    csv = rospy.get_param("~stop_classes_csv", "person,car")
    return [c.strip().lower() for c in csv.split(",") if c.strip()]


def _find_y5_root(p_hint: str) -> Path:
    cands = [
        Path(p_hint).expanduser().resolve(),
        Path.cwd() / "yolov5",
        Path(__file__).resolve().parent / "yolov5",
    ]
    for r in cands:
        if (r / "models" / "common.py").exists() and (r / "utils" / "general.py").exists():
            return r
    raise RuntimeError("yolov5 디렉토리를 찾지 못했습니다. ~yolov5_dir 또는 ./yolov5 확인.")


def _resolve_weights(y5root: Path, weights: str) -> str:
    p = Path(weights)
    if p.is_absolute() and p.exists():
        return str(p)
    if (y5root / weights).exists():
        return str(y5root / weights)
    if p.exists():
        return str(p)
    raise RuntimeError(f"가중치 파일을 찾지 못했습니다: {weights}")


class YoloObstacleStopNode:
    def __init__(self):
        # ───────── 입력/출력 토픽 ─────────
        self.image_topic      = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.emergency_topic  = rospy.get_param("~emergency_topic", "/emergency_stop")
        self.publish_cmd_vel  = bool(rospy.get_param("~publish_cmd_vel", False))
        self.cmd_vel_topic    = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # ───────── YOLO 설정 ─────────
        self.yolov5_dir = rospy.get_param("~yolov5_dir", f"/home/{os.getenv('USER','user')}/catkin_ws/src/yolov5")
        self.weights    = rospy.get_param("~weights", "yolov5s.pt")
        self.imgsz      = int(rospy.get_param("~imgsz", 640))
        self.conf_thres = float(rospy.get_param("~conf_thres", 0.40))
        self.iou_thres  = float(rospy.get_param("~iou_thres", 0.45))
        self.max_det    = int(rospy.get_param("~max_det", 50))

        # ───────── 침입 판정 파라미터 ─────────
        self.stop_classes = _parse_stop_classes()           # 기본 ["person","car"]
        self.roi_bottom_ratio  = float(rospy.get_param("~roi_bottom_ratio", 0.35))
        self.roi_center_ratio  = float(rospy.get_param("~roi_center_ratio", 0.8))
        self.min_box_area_ratio = float(rospy.get_param("~min_box_area_ratio", 0.004))  # 0.4% 이상
        self.min_conf_override  = float(rospy.get_param("~min_conf_override", 0.0))     # 0이면 무시

        # ───────── 투표/유지/쿨다운 ─────────
        self.vote_window    = int(rospy.get_param("~vote_window", 5))
        self.vote_threshold = int(rospy.get_param("~vote_threshold", 3))
        self.stop_hold_sec  = float(rospy.get_param("~stop_hold_sec", 1.5))
        self.cooldown_sec   = float(rospy.get_param("~cooldown_sec", 0.8))

        # ───────── 디버그 ─────────
        self.show_debug  = bool(rospy.get_param("~show_debug", True))
        self.window_name = rospy.get_param("~window_name", "YOLO Obstacle Stop Debug")

        # ───────── YOLO import/load ─────────
        y5root = _find_y5_root(self.yolov5_dir)
        if str(y5root) not in sys.path:
            sys.path.insert(0, str(y5root))
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox

        self.DetectMultiBackend = DetectMultiBackend
        self.nms = non_max_suppression
        self.scale_boxes = scale_boxes
        self.letterbox = letterbox

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half   = (self.device.type != 'cpu')
        _logi(f"[YOLO] device={self.device}, fp16={self.half}")

        weights_path = _resolve_weights(y5root, self.weights)
        self.model = self.DetectMultiBackend(weights_path, device=self.device, dnn=False, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        try:
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        except Exception:
            pass

        # 클래스 라벨 표준화(dict→list 보정)
        if isinstance(self.names, dict):
            mx = max(int(k) for k in self.names.keys())
            tmp = [""] * (mx + 1)
            for k, v in self.names.items():
                tmp[int(k)] = v
            self.names = tmp

        # 이름 → id 맵
        self.stop_class_ids = {i for i, n in enumerate(self.names) if n and n.strip().lower() in self.stop_classes}
        if not self.stop_class_ids:
            _logw(f"[YOLO] stop_classes={self.stop_classes} 에 해당하는 라벨이 모델에 없습니다. → 모든 클래스를 대상으로 함")
            self.stop_class_ids = None  # None이면 전체 허용

        # ───────── ROS I/O ─────────
        self.bridge    = CvBridge()
        self.pub_stop  = rospy.Publisher(self.emergency_topic, Bool, queue_size=10)
        self.pub_twist = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10) if self.publish_cmd_vel else None
        self.sub_img   = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)

        # ───────── 상태 ─────────
        self.votes = deque(maxlen=self.vote_window)
        self.stop_active = False
        self.stop_until = 0.0
        self.cooldown_until = 0.0

        _logi(f"[YOLOStop] Ready. image_topic={self.image_topic}, emergency_topic={self.emergency_topic}, "
              f"ROI(bottom={self.roi_bottom_ratio*100:.0f}%, center={self.roi_center_ratio*100:.0f}%), "
              f"min_box_area_ratio={self.min_box_area_ratio:.4f}, classes={sorted(self.stop_classes)}")

    @torch.no_grad()
    def image_cb(self, msg: Image):
        # 1) 이미지 획득
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            _logw(f"[YOLOStop] cv_bridge failed: {e}")
            return

        H, W = bgr.shape[:2]
        # 2) ROI 정의 (하단 & 중앙 폭)
        top = int(H * (1.0 - self.roi_bottom_ratio))
        top = max(0, min(H-1, top))
        if self.roi_center_ratio < 1.0:
            cw = int(W * self.roi_center_ratio)
            x0 = max(0, (W - cw) // 2)
            x1 = min(W, x0 + cw)
        else:
            x0, x1 = 0, W

        # 3) YOLO 추론
        im = self.letterbox(bgr, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        t = torch.from_numpy(im).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0
        if t.ndimension() == 3:
            t = t.unsqueeze(0)

        pred = self.model(t, augment=False)
        pred = self.nms(pred, self.conf_thres, self.iou_thres, classes=None, max_det=self.max_det)

        intruder = False
        min_conf = max(0.0, float(self.min_conf_override)) if self.min_conf_override > 0.0 else self.conf_thres
        vis = bgr

        for det in pred:
            if len(det):
                det[:, :4] = self.scale_boxes(t.shape[2:], det[:, :4], bgr.shape).round()
                for *xyxy, conf, cls in det:
                    cls_i = int(cls.item())
                    if (self.stop_class_ids is not None) and (cls_i not in self.stop_class_ids):
                        continue
                    conf_f = float(conf.item())
                    if conf_f < min_conf:
                        continue

                    x1b, y1b, x2b, y2b = [int(v.item()) for v in xyxy]
                    x1b, y1b = max(0, x1b), max(0, y1b)
                    x2b, y2b = min(W-1, x2b), min(H-1, y2b)
                    if x2b <= x1b or y2b <= y1b:
                        continue

                    # 박스 면적 비율로 먼/작은 객체 배제
                    box_area = (x2b - x1b) * (y2b - y1b)
                    area_ratio = box_area / float(W * H)
                    if area_ratio < self.min_box_area_ratio:
                        draw_color = (128, 128, 128)  # 너무 작음(무시)
                        if self.show_debug:
                            cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), draw_color, 1)
                        continue

                    # ROI 교차 확인(하단+중앙 영역과 교집합이 있을 때만 침입 후보)
                    inter_x1 = max(x1b, x0)
                    inter_y1 = max(y1b, top)
                    inter_x2 = min(x2b, x1)
                    inter_y2 = min(y2b, H)
                    intersects_roi = (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

                    draw_color = (0, 165, 255) if intersects_roi else (255, 255, 0)  # ROI내=주황, ROI밖=하늘색
                    if self.show_debug:
                        name = self.names[cls_i] if cls_i < len(self.names) else str(cls_i)
                        cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), draw_color, 2)
                        cv2.putText(vis, f"{name} {conf_f:.2f} a={area_ratio*100:.1f}%",
                                    (x1b, max(0, y1b-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2, cv2.LINE_AA)

                    if intersects_roi:
                        intruder = True

        # 4) 투표/트리거/해제
        self.votes.append(1 if intruder else 0)
        vote_sum = sum(self.votes)
        vote_ok = (vote_sum >= self.vote_threshold)
        now = ptime.monotonic()

        # 트리거
        if (not self.stop_active) and (now >= self.cooldown_until) and vote_ok:
            self.stop_active = True
            self.stop_until = now + self.stop_hold_sec
            self.publish_stop(True)
            _logw(f"[YOLOStop] EMERGENCY STOP TRIGGERED (votes={vote_sum}/{self.vote_window})")

        # 유지/해제
        if self.stop_active:
            if self.publish_cmd_vel:
                self.publish_zero_twist()
            if now >= self.stop_until:
                self.stop_active = False
                self.cooldown_until = now + self.cooldown_sec
                self.publish_stop(False)
                _logi(f"[YOLOStop] Stop released (cooldown {self.cooldown_sec:.1f}s)")

        # 5) 디버그 뷰
        if self.show_debug:
            try:
                # ROI 시각화
                cv2.rectangle(vis, (x0, top), (x1-1, H-1), (0, 255, 255), 2)
                status = "INTRUDER" if intruder else "clear"
                color = (0, 0, 255) if intruder else (0, 255, 0)
                cv2.putText(vis, f"votes={vote_sum}/{self.vote_window}  {status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.imshow(self.window_name, vis)
                cv2.waitKey(1)
            except Exception:
                pass  # 헤드리스 환경

    # Bool 퍼블리시(+ 옵션: 즉시 0속도 1회)
    def publish_stop(self, flag: bool):
        self.pub_stop.publish(Bool(data=flag))
        if flag and self.publish_cmd_vel:
            self.publish_zero_twist()

    # (옵션) 0속도 트위스트 1회
    def publish_zero_twist(self):
        if self.pub_twist is None:
            return
        self.pub_twist.publish(Twist())  # all zeros


def main():
    rospy.init_node("yolov5_obstacle_stop_node", anonymous=False)
    try:
        YoloObstacleStopNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

