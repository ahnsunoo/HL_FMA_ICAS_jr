	#!/usr/bin/env python3
	# -*- coding: utf-8 -*-
	"""
	Webcam/ROS 이미지 입력 + YOLOv5 기반 신호등 판별 노드
	- 입력 모드 선택: (1) 일반 웹캠(OpenCV)  (2) ROS Image 토픽
	- 색 판별: 좌/우 2분할(왼쪽=RED, 오른쪽=GREEN), HSV 비율 기반 (선택적으로 Lab ΔE 크로스체크)
	- 검출 박스: 가로형(landscape)만 사용 옵션, 다중 검출 시 선택정책 지원
	- 최종 판정: '최근 vote_time_window_sec' 동안의 시간 비율(녹색>빨강>unknown 우선)
	- 판정 갱신 주기: decide_period_sec 마다 최종판정 업데이트
	- 출력: /traffic_light_state(String: 'green'|'red'|'unknown'), /traffic_light_stop(Bool: 빨강이면 True)

	[준비]
	- YOLOv5 로컬 클론 및 가중치(pt)
	- ROS + OpenCV + PyTorch 설치 환경
	- (웹캠 모드) USB 카메라 연결
	- (ROS 이미지 모드) 카메라 토픽(/camera/color/image_raw 등) 발행 중

	[실행 예]
	rosrun your_pkg webcam_or_rosimg_yolov5_tl.py \
	  _input_mode:=webcam _cam_index:=0 _cap_width:=1280 _cap_height:=720 \
	  _yolov5_dir:=/home/USER/catkin_ws/src/yolov5 _weights:=/home/USER/catkin_ws/src/yolov5/yolov5s.pt \
	  _vote_time_window_sec:=2.0 _decide_period_sec:=0.5 \
	  _selection_mode:=center_then_largest _enable_horizontal_only:=true _horiz_ar_min:=1.3
	"""

	# ─────────────────────────────────────────────────────────────────────────────
	#                          ▲ 최 상 단  설 정  (기본값) ▲
	#               (rosparam "~키"로 런타임 덮어쓰기 가능)
	# ─────────────────────────────────────────────────────────────────────────────
	import os as _os
	DEFAULTS = {
	    # I/O (토픽/루프/판정주기)
	    "status_topic": "/traffic_light_state",
	    "stop_topic": "/traffic_light_stop",
	    "rate_hz": 30.0,
	    "vote_time_window_sec": 2.0,    # 최근 N초 히스토리로 최종판정
	    "decide_period_sec": 0.5,       # 판정 갱신 주기
	    "history_sec": 10.0,            # 히스토리 보존 상한(여유 버퍼)

	    # 입력 모드: "webcam" | "ros_image"
	    "input_mode": "webcam",

	    # (웹캠) 캡처
	    "cam_index": 0,
	    "cap_width": 0,                 # 0이면 드라이버 기본
	    "cap_height": 0,
	    "cap_fps": 0,

	    # (ROS 이미지) 토픽
	    "image_topic": "/camera/color/image_raw",

	    # YOLOv5 경로/가중치/추론
	    "yolov5_dir": f"/home/{_os.getenv('USER','user')}/catkin_ws/src/yolov5",
	    "weights": "yolov5m.pt",
	    "imgsz": 640,
	    "conf_thres": 0.35,
	    "iou_thres": 0.45,
	    "max_det": 20,

	    # 디버그
	    "show_debug": True,

	    # 박스 필터/선택 정책
	    "enable_horizontal_only": True, # 가로형 박스만 사용
	    "horiz_ar_min": 1.3,            # 가로/세로비 하한
	    "min_box_w": 16,                # 박스 최소 너비/높이
	    "min_box_h": 16,
	    # selection_mode: center_then_largest | largest | highest_conf | leftmost | rightmost | center_closest | hybrid
	    "selection_mode": "center_then_largest",
	    "hybrid_center_weight": 0.4,    # hybrid: 중심 가중
	    "hybrid_area_weight": 0.6,      # hybrid: 면적 가중

	    # 좌/우 ROI
	    "edge_crop_ratio": 0.03,        # 좌우 3% 크롭 후 좌/우 반분
	    "morph_kernel": 3,              # HSV 마스크 모폴로지(노이즈 완화) 커널 크기(0=비활성)

	    # HSV 임계 (OpenCV 스케일: H∈[0,180], S,V∈[0,255])
	    "sat_min": 55,
	    "val_min": 100,
	    "green_h_lo": 74, "green_h_hi": 98,         # Green 범위
	    "red_h_lo1": 0,  "red_h_hi1": 12,           # Red 랩어라운드 ↓
	    "red_h_lo2": 168,"red_h_hi2": 180,

	    # 활성 비율 임계(ROI 내 픽셀 비율)
	    "thr_green_ratio": 0.010,
	    "thr_red_ratio":   0.010,

	    # 저조도 보정(CLAHE + 감마)
	    "gamma": 0.88,                  # <1 밝게
	    "clahe_clip": 2.0,
	    "clahe_tile": 8,

	    # (선택) Lab ΔE 크로스체크(fallback) — 기본 비활성
	    "enable_lab_crosscheck": False,
	    "lab_thr_green": 30.0,
	    "lab_thr_red":   32.0,
	}

	# ─────────────────────────────────────────────────────────────────────────────
	import sys
	import time
	import math
	import collections
	from pathlib import Path
	from typing import Tuple

	import cv2
	import numpy as np
	import torch
	import rospy
	from std_msgs.msg import String, Bool

	# (ROS 이미지 모드일 때만 필요)
	try:
	    from cv_bridge import CvBridge
	    from sensor_msgs.msg import Image
	except Exception:
	    CvBridge, Image = None, None


	def _P(key: str):
	    return rospy.get_param(f"~{key}", DEFAULTS[key])


	class YoloV5TrafficLightNode:
	    """웹캠/ROS Image → YOLOv5 검출 → (가로형 박스 필터, 다중검출 선택정책) → 좌/우 2분할 HSV 비율 판정
	       → 최근 N초 시간가중 투표(녹색>빨강>unknown) → 주기적 최종판정 퍼블리시"""

	    # ──────────────────────────────────────────────────
	    # 초기화
	    # ──────────────────────────────────────────────────
	    def __init__(self):
		# 공통 파라미터
		self.status_topic   = _P("status_topic")
		self.stop_topic     = _P("stop_topic")
		self.rate_hz        = float(_P("rate_hz"))
		self.vote_time_window_sec = float(_P("vote_time_window_sec"))
		self.decide_period_sec    = float(_P("decide_period_sec"))
		self.history_sec    = float(_P("history_sec"))
		self.input_mode     = str(_P("input_mode")).strip().lower()
		self.show_debug     = bool(_P("show_debug"))

		# 입력 파라미터
		self.cam_index      = int(_P("cam_index"))
		self.cap_width      = int(_P("cap_width"))
		self.cap_height     = int(_P("cap_height"))
		self.cap_fps        = int(_P("cap_fps"))
		self.image_topic    = _P("image_topic")

		# YOLO 설정
		self.yolov5_dir     = _P("yolov5_dir")
		self.weights_path   = rospy.get_param("~weights", str(Path(self.yolov5_dir) / DEFAULTS["weights"]))
		self.imgsz          = int(_P("imgsz"))
		self.conf_thres     = float(_P("conf_thres"))
		self.iou_thres      = float(_P("iou_thres"))
		self.max_det        = int(_P("max_det"))

		# 박스/정책
		self.enable_horizontal_only = bool(_P("enable_horizontal_only"))
		self.horiz_ar_min           = float(_P("horiz_ar_min"))
		self.min_box_w              = int(_P("min_box_w"))
		self.min_box_h              = int(_P("min_box_h"))
		self.selection_mode         = str(_P("selection_mode")).strip().lower()
		self.hybrid_center_weight   = float(_P("hybrid_center_weight"))
		self.hybrid_area_weight     = float(_P("hybrid_area_weight"))

		# ROI/HSV/저조도
		self.edge_crop_ratio = float(_P("edge_crop_ratio"))
		self.morph_kernel    = int(_P("morph_kernel"))
		self.sat_min         = int(_P("sat_min"))
		self.val_min         = int(_P("val_min"))
		self.green_h_lo      = int(_P("green_h_lo")); self.green_h_hi = int(_P("green_h_hi"))
		self.red_h_lo1       = int(_P("red_h_lo1"));  self.red_h_hi1  = int(_P("red_h_hi1"))
		self.red_h_lo2       = int(_P("red_h_lo2"));  self.red_h_hi2  = int(_P("red_h_hi2"))
		self.thr_green_ratio = float(_P("thr_green_ratio"))
		self.thr_red_ratio   = float(_P("thr_red_ratio"))
		self.gamma           = float(_P("gamma"))
		self.clahe_clip      = float(_P("clahe_clip"))
		self.clahe_tile      = int(_P("clahe_tile"))

		# Lab ΔE(선택)
		self.enable_lab_crosscheck = bool(_P("enable_lab_crosscheck"))
		self.lab_thr_green     = float(_P("lab_thr_green"))
		self.lab_thr_red       = float(_P("lab_thr_red"))

		# 시간 히스토리: (t_sec, 'green'|'red'|'unknown')
		self.state_hist = collections.deque(maxlen=int(max(1000, self.history_sec * self.rate_hz * 2)))
		self.last_decide_time = 0.0
		self.last_final_state = "unknown"

		# 퍼블리셔
		self.pub_state = rospy.Publisher(self.status_topic, String, queue_size=10)
		self.pub_stop  = rospy.Publisher(self.stop_topic,   Bool,   queue_size=10)

		# YOLOv5 모듈 불러오기/모델 로드
		self._init_yolov5()

		# 입력 초기화
		self.bridge = None
		self.cap    = None
		self.last_vis = None

		if self.input_mode == "webcam":
		    self._init_webcam()
		elif self.input_mode == "ros_image":
		    self._init_ros_image()
		else:
		    raise RuntimeError(f"Unknown ~input_mode='{self.input_mode}' (use 'webcam' or 'ros_image')")

		# (ROS 이미지 모드에서) 주기적 최종판정 업데이터
		if self.input_mode == "ros_image":
		    self._timer = rospy.Timer(rospy.Duration(self.decide_period_sec), self._timer_publish)

		rospy.loginfo("[READY] YOLOv5 TL (HSV 2-split, landscape filter, policy, time-window voting)")

	    # ──────────────────────────────────────────────────
	    # YOLO 초기화
	    # ──────────────────────────────────────────────────
	    def _init_yolov5(self):
		y5_root = Path(self.yolov5_dir).resolve()
		if not (y5_root / "models" / "common.py").exists() or not (y5_root / "utils" / "general.py").exists():
		    rospy.logerr(f"[YOLOv5] wrong yolov5_dir: {y5_root}")
		    raise RuntimeError("yolov5_dir must contain models/common.py and utils/general.py")
		if str(y5_root) not in sys.path:
		    sys.path.insert(0, str(y5_root))

		try:
		    from models.common import DetectMultiBackend
		    from utils.general import non_max_suppression, scale_boxes
		    from utils.augmentations import letterbox
		    import models.common as _mc
		    import utils as _ut
		    rospy.loginfo(f"[YOLOv5] models.common: {_mc.__file__}")
		    rospy.loginfo(f"[YOLOv5] utils       : {_ut.__file__}")
		except Exception as e:
		    rospy.logerr(f"[YOLOv5] import failed: {e}")
		    raise

		self.DetectMultiBackend  = DetectMultiBackend
		self.non_max_suppression = non_max_suppression
		self.scale_boxes         = scale_boxes
		self.letterbox           = letterbox

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.half   = (self.device.type != 'cpu')
		rospy.loginfo(f"[INFO] device={self.device}, fp16={self.half}")

		self.model  = self.DetectMultiBackend(self.weights_path, device=self.device, dnn=False, fp16=self.half)
		self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

		try:
		    self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
		except Exception:
		    pass

		# 클래스 이름 정리
		if isinstance(self.names, dict):
		    mx = max(int(k) for k in self.names.keys())
		    tmp = [""] * (mx + 1)
		    for k, v in self.names.items():
		        tmp[int(k)] = v
		    self.names = tmp

		# 'traffic light' 클래스 id
		self.tl_class_ids = [i for i, n in enumerate(self.names) if n and n.strip().lower() == "traffic light"]
		if not self.tl_class_ids:
		    rospy.logwarn("[WARN] 'traffic light' class not found in model names. No class filter applied.")
		    self.tl_class_ids = None
		else:
		    rospy.loginfo(f"[INFO] traffic light class ids: {self.tl_class_ids}")

	    # ──────────────────────────────────────────────────
	    # 입력 초기화(웹캠/ROS 이미지)
	    # ──────────────────────────────────────────────────
	    def _init_webcam(self):
		backend = cv2.CAP_DSHOW if _os.name == 'nt' else 0
		self.cap = cv2.VideoCapture(self.cam_index, backend)
		if self.cap_width  > 0: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_width)
		if self.cap_height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
		if self.cap_fps    > 0: self.cap.set(cv2.CAP_PROP_FPS,          self.cap_fps)
		if not self.cap.isOpened():
		    raise RuntimeError(f"Webcam open failed at index {self.cam_index}")

	    def _init_ros_image(self):
		if CvBridge is None or Image is None:
		    raise RuntimeError("cv_bridge or sensor_msgs/Image not available")
		self.bridge = CvBridge()
		self.sub_img = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24)

	    # ──────────────────────────────────────────────────
	    # 공통 유틸
	    # ──────────────────────────────────────────────────
	    def _now_sec(self) -> float:
		t = rospy.get_time()
		return t if t > 0 else time.time()

	    def _append_state(self, t_sec: float, state: str):
		self.state_hist.append((t_sec, state))

	    def _trim_history(self, now_sec: float):
		cutoff = now_sec - max(self.history_sec, self.vote_time_window_sec * 3.0)
		while self.state_hist and self.state_hist[0][0] < cutoff:
		    self.state_hist.popleft()

	    def _time_weighted_winner(self, now_sec: float) -> str:
		"""최근 vote_time_window_sec 동안의 '시간'을 합산하여 최종 상태 선택.
		   동률 시 우선순위: green > red > unknown"""
		if not self.state_hist:
		    return "unknown"
		window = self.vote_time_window_sec
		cutoff = now_sec - window
		events = list(self.state_hist)

		# 창 시작 시점의 상태 찾기
		idx = len(events) - 1
		while idx >= 0 and events[idx][0] > cutoff:
		    idx -= 1
		state_at_cut = events[idx][1] if idx >= 0 else events[0][1]

		segs = [(cutoff, state_at_cut)]
		for t, s in events:
		    if cutoff <= t <= now_sec:
		        segs.append((t, s))
		last_state = segs[-1][1] if segs else events[-1][1]
		segs.append((now_sec, last_state))

		acc = {"green": 0.0, "red": 0.0, "unknown": 0.0}
		for i in range(len(segs)-1):
		    t0, s0 = segs[i]
		    t1, _  = segs[i+1]
		    dt = max(0.0, t1 - t0)
		    acc[s0 if s0 in acc else "unknown"] += dt

		prio = {"green": 2, "red": 1, "unknown": 0}
		return max(acc.items(), key=lambda kv: (kv[1], prio.get(kv[0], -1)))[0]

	    # ──────────────────────────────────────────────────
	    # 메인 루프(웹캠)
	    # ──────────────────────────────────────────────────
	    def spin(self):
		if self.input_mode == "ros_image":
		    # ROS 이미지 모드는 콜백+타이머로 처리
		    rospy.spin()
		    return

		rate = rospy.Rate(self.rate_hz)
		while not rospy.is_shutdown():
		    ok, bgr = self.cap.read()
		    if not ok or bgr is None:
		        rospy.logwarn("[Camera] Empty frame")
		        rate.sleep()
		        continue

		    instant_state, vis = self._process_one_frame(bgr)
		    now = self._now_sec()
		    self._append_state(now, instant_state)
		    self._trim_history(now)

		    # 주기적으로 최종판정 갱신
		    if (now - self.last_decide_time) >= self.decide_period_sec:
		        self.last_final_state = self._time_weighted_winner(now)
		        self.last_decide_time = now
		        self.pub_state.publish(String(data=self.last_final_state))
		        self.pub_stop.publish(Bool(data=(self.last_final_state == "red")))

		    if self.show_debug:
		        try:
		            vis = self._draw_status(vis, self.last_final_state)
		            cv2.imshow("Webcam | YOLOv5 TL", vis)
		            cv2.waitKey(1)
		        except Exception:
		            pass

		    rate.sleep()

	    # ──────────────────────────────────────────────────
	    # ROS 이미지 콜백 + 타이머 퍼블리시
	    # ──────────────────────────────────────────────────
	    @torch.no_grad()
	    def _image_cb(self, msg: Image):
		try:
		    bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except Exception as e:
		    rospy.logwarn(f"[cv_bridge] Image conversion failed: {e}")
		    return

		instant_state, vis = self._process_one_frame(bgr)
		self.last_vis = vis
		now = self._now_sec()
		self._append_state(now, instant_state)
		self._trim_history(now)

		if self.show_debug:
		    try:
		        vis = self._draw_status(vis, self.last_final_state)
		        cv2.imshow("ROS Image | YOLOv5 TL", vis)
		        cv2.waitKey(1)
		    except Exception:
		        pass

	    def _timer_publish(self, _evt):
		now = self._now_sec()
		self.last_final_state = self._time_weighted_winner(now)
		self.last_decide_time = now
		self.pub_state.publish(String(data=self.last_final_state))
		self.pub_stop.publish(Bool(data=(self.last_final_state == "red")))

	    # ──────────────────────────────────────────────────
	    # 1프레임 처리: YOLO → 후보필터/정책 → 좌/우 2분할 HSV 판정
	    # ──────────────────────────────────────────────────
	    @torch.no_grad()
	    def _process_one_frame(self, bgr: np.ndarray):
		im0 = bgr.copy()
		H, W = im0.shape[:2]

		# letterbox → tensor
		im = self.letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]
		im = im.transpose((2, 0, 1))[::-1]
		im = np.ascontiguousarray(im)

		t = torch.from_numpy(im).to(self.device)
		t = t.half() if self.half else t.float()
		t /= 255.0
		if t.ndimension() == 3:
		    t = t.unsqueeze(0)

		pred = self.model(t, augment=False)
		pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, max_det=self.max_det)

		candidates = []  # (x1,y1,x2,y2, conf, cls, dist2, area)
		cx_img, cy_img = W*0.5, H*0.5

		for det in pred:
		    if len(det):
		        det[:, :4] = self.scale_boxes(t.shape[2:], det[:, :4], im0.shape).round()
		        for *xyxy, conf, cls in det:
		            cls_i = int(cls.item())
		            if (self.tl_class_ids is not None) and (cls_i not in self.tl_class_ids):
		                continue
		            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
		            x1, y1 = max(0, x1), max(0, y1)
		            x2, y2 = min(W-1, x2), min(H-1, y2)
		            if x2 <= x1 or y2 <= y1:
		                continue
		            w, h = (x2-x1), (y2-y1)
		            if w < self.min_box_w or h < self.min_box_h:
		                continue
		            if self.enable_horizontal_only:
		                ar = w/float(h) if h>0 else 0
		                if ar < self.horiz_ar_min:
		                    continue
		            bx, by = 0.5*(x1+x2), 0.5*(y1+y2)
		            dist2 = (bx-cx_img)**2 + (by-cy_img)**2
		            area = w*h
		            candidates.append((x1,y1,x2,y2,float(conf.item()),cls_i,dist2,area))

		if not candidates:
		    return "unknown", im0

		x1,y1,x2,y2,conf,cls,_,_ = self._select_box(candidates, W)

		crop = im0[y1:y2, x1:x2]
		state = self._classify_lr_hsv(crop)  # 'green'|'red'|'unknown'

		# 디버그 그리기
		draw_color = (0,255,0) if state=="green" else (0,0,255) if state=="red" else (0,255,255)
		cv2.rectangle(im0, (x1,y1), (x2,y2), draw_color, 2)
		label_name = self.names[cls] if cls < len(self.names) else 'tl'
		cv2.putText(im0, f"{label_name} {conf:.2f} {state} | {self.selection_mode}", (x1, max(0, y1-6)),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2, cv2.LINE_AA)

		return state, im0

	    # ──────────────────────────────────────────────────
	    # 다중검출 선택 정책
	    # ──────────────────────────────────────────────────
	    def _select_box(self, cands, W_img):
		mode = self.selection_mode
		if mode == "largest":
		    c = max(cands, key=lambda c: c[7])                 # area
		elif mode == "highest_conf":
		    c = max(cands, key=lambda c: c[4])                 # conf
		elif mode == "leftmost":
		    c = min(cands, key=lambda c: 0.5*(c[0]+c[2]))      # center x
		elif mode == "rightmost":
		    c = max(cands, key=lambda c: 0.5*(c[0]+c[2]))
		elif mode == "center_closest":
		    c = min(cands, key=lambda c: c[6])                 # dist2
		elif mode == "hybrid":
		    # score = w_center*(norm dist) - w_area*(norm area) → 작은게 우수
		    max_area  = max(x[7] for x in cands) or 1.0
		    max_dist2 = max(x[6] for x in cands) or 1.0
		    wc, wa = max(0.0, self.hybrid_center_weight), max(0.0, self.hybrid_area_weight)
		    scored = []
		    for x in cands:
		        nd = x[6]/max_dist2
		        na = x[7]/max_area
		        score = wc*nd - wa*na
		        scored.append((score, x))
		    c = min(scored, key=lambda s: (s[0], -s[1][7]))[1]
		else:
		    # center_then_largest
		    cands.sort(key=lambda c: (c[6], -c[7]))
		    c = cands[0]
		return c

	    # ──────────────────────────────────────────────────
	    # 좌/우 2분할(왼쪽=RED, 오른쪽=GREEN) + HSV 비율 판정 (Lab ΔE 선택적 보강)
	    # 우선순위: green > red
	    # ──────────────────────────────────────────────────
	    def _classify_lr_hsv(self, bgr: np.ndarray) -> str:
		if bgr is None or bgr.size == 0:
		    return "unknown"

		# 저조도 보정
		bgr = self._preprocess_luma(bgr)

		# 좌/우 분할
		left, right = self._split_left_right(bgr)

		# HSV 비율
		r_ratio = self._ratio_red(left)
		g_ratio = self._ratio_green(right)

		# (선택) Lab ΔE 크로스체크로 희미한 신호 구제
		if self.enable_lab_crosscheck:
		    if g_ratio < self.thr_green_ratio:
		        g_ratio = max(g_ratio, self._ratio_green_lab(right))
		    if r_ratio < self.thr_red_ratio:
		        r_ratio = max(r_ratio, self._ratio_red_lab(left))

		green_active = (g_ratio >= self.thr_green_ratio)
		red_active   = (r_ratio >= self.thr_red_ratio)

		if green_active:
		    return "green"
		elif red_active:
		    return "red"
		else:
		    return "unknown"

	    # 좌우 반분(양쪽 3% 크롭 후 반분)
	    def _split_left_right(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		H, W = bgr.shape[:2]
		if H < 4 or W < 4:
		    return bgr, bgr
		pad = int(self.edge_crop_ratio * W)
		x0, x1 = max(0, pad), min(W-pad, W)
		roi = bgr[:, x0:x1]
		mid = roi.shape[1] // 2
		return roi[:, :mid], roi[:, mid:]

	    # HSV 마스크 전처리(옵션: 모폴로지 오프닝)
	    def _post_mask(self, mask: np.ndarray) -> np.ndarray:
		if self.morph_kernel and self.morph_kernel >= 2:
		    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
		    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
		return mask

	    # GREEN 비율
	    def _ratio_green(self, bgr: np.ndarray) -> float:
		if bgr is None or bgr.size == 0:
		    return 0.0
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		H,S,V = cv2.split(hsv)
		base = (S >= self.sat_min) & (V >= self.val_min)
		gmask = base & (H >= self.green_h_lo) & (H <= self.green_h_hi)
		gmask = self._post_mask(gmask.astype(np.uint8))
		return float(np.count_nonzero(gmask)) / max(1.0, float(H.size))

	    # RED 비율(랩어라운드)
	    def _ratio_red(self, bgr: np.ndarray) -> float:
		if bgr is None or bgr.size == 0:
		    return 0.0
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		H,S,V = cv2.split(hsv)
		base = (S >= self.sat_min) & (V >= self.val_min)
		red1 = base & (H >= self.red_h_lo1) & (H <= self.red_h_hi1)
		red2 = base & (H >= self.red_h_lo2) & (H <= self.red_h_hi2)
		rmask = (red1 | red2).astype(np.uint8)
		rmask = self._post_mask(rmask)
		return float(np.count_nonzero(rmask)) / max(1.0, float(H.size))

	    # ── (선택) Lab ΔE 크로스체크: 희미한 신호 구제용 ──
	    def _ratio_green_lab(self, bgr: np.ndarray) -> float:
		if bgr is None or bgr.size == 0:
		    return 0.0
		lab = cv2.cvtColor(self._downscale(self._preprocess_luma(bgr)), cv2.COLOR_BGR2LAB).astype(np.float32)
		L,A,B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
		gr = self._get_lab_refs()["green"]
		de = self._deltaE76_min(L,A,B, gr)
		V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float32)
		mask = (de <= self.lab_thr_green) & (V >= self.val_min)  # 너무 어두우면 제외
		return float(np.count_nonzero(mask)) / max(1.0, float(L.size))

	    def _ratio_red_lab(self, bgr: np.ndarray) -> float:
		if bgr is None or bgr.size == 0:
		    return 0.0
		lab = cv2.cvtColor(self._downscale(self._preprocess_luma(bgr)), cv2.COLOR_BGR2LAB).astype(np.float32)
		L,A,B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
		rb, rd = self._get_lab_refs()["red_bright"], self._get_lab_refs()["red_dark"]
		de = np.minimum(self._deltaE76_min(L,A,B, rb), self._deltaE76_min(L,A,B, rd))
		V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float32)
		mask = (de <= self.lab_thr_red) & (V >= self.val_min)
		return float(np.count_nonzero(mask)) / max(1.0, float(L.size))

	    def _downscale(self, img: np.ndarray) -> np.ndarray:
		h,w = img.shape[:2]
		s = max(1, int(max(h,w)/64))
		return cv2.resize(img, (w//s, h//s), interpolation=cv2.INTER_AREA) if s>1 else img

	    def _deltaE76_min(self, L,A,B, ref_mat):
		dL = L[None,:,:] - ref_mat[:,0][:,None,None]
		dA = A[None,:,:] - ref_mat[:,1][:,None,None]
		dB = B[None,:,:] - ref_mat[:,2][:,None,None]
		de = np.sqrt(dL*dL + dA*dA + dB*dB)
		return np.min(de, axis=0)

	    def _get_lab_refs(self):
		if hasattr(self, "_lab_refs"):
		    return self._lab_refs

		def jitter_hex(hex_code, l_delta=(-8,0,+8), s_delta=(-10,0,+10)):
		    bgr = np.uint8([[list(self._hex_to_bgr(hex_code))]])
		    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)[0,0]
		    H0,S0,V0 = int(hsv[0]), int(hsv[1]), int(hsv[2])
		    samples=[]
		    for dv in l_delta:
		        for ds in s_delta:
		            H,S,V = H0, np.clip(S0+ds,0,255), np.clip(V0+dv,0,255)
		            hsv1 = np.uint8([[[H,S,V]]])
		            bgr1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)[0,0]
		            lab1 = cv2.cvtColor(np.uint8([[bgr1]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0,0]
		            samples.append(lab1)
		    return np.stack(samples, axis=0)

		self._lab_refs = {
		    "red_bright": jitter_hex("#df111b"),
		    "red_dark":   jitter_hex("#4d1216"),
		    "green":      jitter_hex("#10a08e"),
		}
		return self._lab_refs

	    def _hex_to_bgr(self, hex_code: str):
		hex_code = hex_code.lstrip('#')
		r = int(hex_code[0:2], 16); g = int(hex_code[2:4], 16); b = int(hex_code[4:6], 16)
		return (b,g,r)

	    # ──────────────────────────────────────────────────
	    # 저조도 대비보정(CLAHE + 감마 하향)
	    # ──────────────────────────────────────────────────
	    def _preprocess_luma(self, bgr: np.ndarray) -> np.ndarray:
		if bgr is None or bgr.size == 0:
		    return bgr
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv)
		clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
		v2 = clahe.apply(v)
		inv = 1.0 / max(self.gamma, 1e-6)
		lut = np.array([((i/255.0)**inv)*255.0 for i in range(256)], dtype=np.float32)
		v3 = cv2.LUT(v2, np.clip(lut,0,255).astype("uint8"))
		return cv2.cvtColor(cv2.merge([h,s,v3]), cv2.COLOR_HSV2BGR)

	    # 디버그 오버레이
	    def _draw_status(self, vis: np.ndarray, state: str) -> np.ndarray:
		if vis is None:
		    return vis
		color = (0,255,0) if state=="green" else (0,0,255) if state=="red" else (0,255,255)
		cv2.putText(vis, f"STATE: {state.upper()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
		info = f"win={self.vote_time_window_sec:.1f}s, period={self.decide_period_sec:.2f}s"
		cv2.putText(vis, info, (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
		return vis


	def main():
	    rospy.init_node("yolov5_traffic_light_node_webcam_or_rosimg")
	    node = None
	    try:
		node = YoloV5TrafficLightNode()
		node.spin()
	    except rospy.ROSInterruptException:
		pass
	    finally:
		try:
		    if node and node.cap:
		        node.cap.release()
		    cv2.destroyAllWindows()
		except Exception:
		    pass


	if __name__ == "__main__":
	    main()

