#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TL + Cone (Parallel/T) Node (Webcam only)
- 신호등/라바콘 항상 추론 + 퍼블리시
- 신호등 결과는 String('red','yellow','rl','green','unknown')로 퍼블리시
- 디버그 시 YOLO 감지 박스 표시(여러 박스 + 대표 박스 강조)

고정 클래스 id:
  red=0, green=1, yellow=2, rl(좌회전)=4

실행 예시(ROS1, catkin):
  source /opt/ros/noetic/setup.bash
  rosrun tl_ctrl webcam.py \
    _cam_index:=0 \
    _cap_width:=640 _cap_height:=640 _cap_fps:=15 \
    _yolov5_dir:=/home/$USER/catkin_ws/src/yolov5 \
    _tl_weights:=/home/icas/catkin_ws/src/tl_ctrl/scripts/weights/tl/best.pt \
    _cone_yolov5_dir:=/home/$USER/catkin_ws/src/yolov5 \
    _cone_weights:=/home/icas/catkin_ws/src/tl_ctrl/scripts/weights/cone/conebest.pt \
    _show_debug:=true \
    _cone_left_x0_ratio:=0.0 _cone_left_x1_ratio:=0.25 \
    _cone_right_x0_ratio:=0.30 _cone_right_x1_ratio:=1.0
"""

import os, sys, time, collections, pathlib
from pathlib import Path
from typing import Tuple, Deque, Dict, List
import cv2, numpy as np, torch, rospy
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse

# ── Windows에서 저장된 .pt를 리눅스에서 로드할 때 경로 객체 호환
try:
    pathlib.WindowsPath = pathlib.PosixPath  # noqa
except Exception:
    pass

# =============================================================================
# ROS 파라미터 기본값 (각 항목에 '쉬운 설명'을 덧붙였습니다)
#  - rosrun/roslaunch 시 _키:=값 으로 덮어쓸 수 있습니다.
# =============================================================================
DEFAULTS = {
    # ── 입력(웹캠) ──────────────────────────────────────────────────────────
    "cam_index": 0,        # 사용할 카메라 인덱스(기본 0). 노트북 내장=0, 외장=1 등
    "cap_width": 640,      # 캡처 폭(px). 640~1280 권장 (너무 크면 FPS 저하)
    "cap_height": 640,     # 캡처 높이(px). YOLO 입력 정사각 권장 시 640 사용
    "cap_fps": 30,         # 요청 FPS. 실제 FPS는 장치/코덱에 따라 달라질 수 있음

    # ── YOLOv5 (신호등) ─────────────────────────────────────────────────────
    "yolov5_dir": "",      # yolov5 루트 디렉터리 경로(예: /home/.../src/yolov5)
    "tl_weights": "",      # 신호등 .pt 가중치 경로(절대경로 또는 yolov5_dir 기준 상대경로)
    "imgsz": 640,          # 신호등 모델 입력 크기(정사각, 보통 640). 속도/정확도 트레이드오프
    "conf_thres": 0.20,    # NMS 전 신뢰도 임계치(낮을수록 박스 많이 통과)
    "iou_thres": 0.25,     # NMS IoU 임계치(겹침 제거 기준). 0.25~0.5 사이 흔함
    "max_det": 50,         # 프레임당 최대 검출 박스 수 제한
    "tl_yolo_min_conf": 0.25,  # ★ 신호등 최종 필터링용 추가 임계치(개별 박스 레벨). conf_thres보다 보수적으로 권장

    # ── 신호등 박스 선택/디스플레이 정책 ────────────────────────────────────
    "min_box_w": 12,       # 무시할 최소 박스 너비(px). 너무 작은 박스(오검출) 제거
    "min_box_h": 12,       # 무시할 최소 박스 높이(px)
    # largest|highest_conf|leftmost|rightmost|center_closest|hybrid
    "selection_mode": "highest_conf",
    "hybrid_center_weight": 0.2,
    "hybrid_area_weight": 0.8,
    "tl_draw_all_boxes": True,
    "tl_box_thickness": 2,

    # ── 신호등 라벨 매핑 ─
    "tl_label_red_csv":    "red",
    "tl_label_green_csv":  "green",
    "tl_label_left_csv":   "rl,left,arrow",
    "tl_label_yellow_csv": "yellow,orange",

    # ── YOLOv5 (라바콘) ─────────────────────────────────────────────────────
    "cone_yolov5_dir": "",
    "cone_weights": "",
    "cone_imgsz": 640,
    "cone_conf_thres": 0.35,
    "cone_iou_thres": 0.45,
    "cone_max_det": 100,
    "cone_label_csv": "cone",

    # ── 라바콘 ROI(관심영역) 설정 ───────────────────────────────────────────
    # 좌측/T: [0.00, 0.25], 우측/평행: [0.30, 1.00]
    "cone_left_x0_ratio": 0.0,
    "cone_left_x1_ratio": 0.25,
    "cone_right_x0_ratio": 0.30,
    "cone_right_x1_ratio": 1.0,
    # 세로 범위(공통)
    "cone_roi_top_ratio": 0.0,
    "cone_roi_bottom_ratio": 1.0,
    # 너무 작은 오검출 제거
    "cone_min_box_area_ratio": 0.0015,

    # ROI 사각형 GUI 표시 여부
    "cone_draw_roi": True,

    # ── 라바콘 서비스 집계(Parallel/T 주차 모드 확정 기준) ────────────────
    "cone_window_sec": 3.0,
    "cone_min_hits": 3,

    # ── 디버그/루프 ─────────────────────────────────────────────────────────
    "show_debug": True,
    "enable_gui": True,
    "rate_hz": 30.0,

    # ── 퍼블리시/투표(신호등 상태 안정화) ───────────────────────────────────
    "status_topic": "/traffic_light_state",
    "stop_topic":   "/red_sign",
    "vote_time_window_sec": 2.0,
    "decide_period_sec":    1.0,
    "history_sec":          10.0,
}

def P(k: str): return rospy.get_param("~" + k, DEFAULTS[k])

def _parse_labels(csv_text: str) -> List[str]:
    return [t.strip().lower() for t in str(csv_text).split(",") if t.strip()]

class TLAndConeNode:
    def __init__(self):
        # 입력/루프
        self.cam_index  = int(P("cam_index"))
        self.cap_width  = int(P("cap_width"))
        self.cap_height = int(P("cap_height"))
        self.cap_fps    = int(P("cap_fps"))
        self.rate_hz    = float(P("rate_hz"))
        self.show_debug = bool(P("show_debug"))
        self.enable_gui = bool(P("enable_gui"))

        # YOLO (신호등)
        self.yolov5_dir   = P("yolov5_dir")
        self.tl_weights   = P("tl_weights")
        self.imgsz        = int(P("imgsz"))
        self.conf_thres   = float(P("conf_thres"))
        self.iou_thres    = float(P("iou_thres"))
        self.max_det      = int(P("max_det"))
        self.tl_yolo_min_conf = float(P("tl_yolo_min_conf"))

        # 디스플레이/선택 정책
        self.min_box_w  = int(P("min_box_w"))
        self.min_box_h  = int(P("min_box_h"))
        self.selection_mode       = str(P("selection_mode")).strip().lower()
        self.hybrid_center_weight = float(P("hybrid_center_weight"))
        self.hybrid_area_weight   = float(P("hybrid_area_weight"))
        self.tl_draw_all_boxes    = bool(P("tl_draw_all_boxes"))
        self.tl_box_thickness     = int(P("tl_box_thickness"))

        # 라벨 CSV
        self.label_map_csv = {
            "red":    _parse_labels(P("tl_label_red_csv")),
            "green":  _parse_labels(P("tl_label_green_csv")),
            "rl":     _parse_labels(P("tl_label_left_csv")),
            "yellow": _parse_labels(P("tl_label_yellow_csv")),
        }

        # 라바콘
        self.cone_y5_dir     = P("cone_yolov5_dir")
        self.cone_weights    = P("cone_weights")
        self.cone_imgsz      = int(P("cone_imgsz"))
        self.cone_conf_thres = float(P("cone_conf_thres"))
        self.cone_iou_thres  = float(P("cone_iou_thres"))
        self.cone_max_det    = int(P("cone_max_det"))
        self.cone_min_box_area_ratio = float(P("cone_min_box_area_ratio"))

        # NEW: 좌/우 ROI 가로 구간(비율), 세로 범위, ROI 표시 여부
        self.cone_left_x0_ratio  = float(P("cone_left_x0_ratio"))
        self.cone_left_x1_ratio  = float(P("cone_left_x1_ratio"))
        self.cone_right_x0_ratio = float(P("cone_right_x0_ratio"))
        self.cone_right_x1_ratio = float(P("cone_right_x1_ratio"))
        self.cone_roi_top_ratio    = float(P("cone_roi_top_ratio"))
        self.cone_roi_bottom_ratio = float(P("cone_roi_bottom_ratio"))
        self.cone_draw_roi       = bool(P("cone_draw_roi"))

        self._hits_parallel: Deque[Tuple[float,int]] = collections.deque(maxlen=4096)
        self._hits_t:        Deque[Tuple[float,int]] = collections.deque(maxlen=4096)

        # 퍼블리셔/투표
        self.status_topic = P("status_topic")
        self.stop_topic   = P("stop_topic")
        self.vote_time_window_sec = float(P("vote_time_window_sec"))
        self.decide_period_sec    = float(P("decide_period_sec"))
        self.history_sec          = float(P("history_sec"))
        self.pub_state   = rospy.Publisher(self.status_topic, String, queue_size=10)
        self.pub_tlcolor = rospy.Publisher(self.stop_topic,   String, queue_size=10)
        self.pub_cone_hit    = rospy.Publisher("/cone_hit", Bool, queue_size=10)
        self.pub_parallel    = rospy.Publisher("/cone_detect_parallel", String, queue_size=1, latch=True)
        self.pub_t           = rospy.Publisher("/cone_detect_t",        String, queue_size=1, latch=True)
        self.pub_parking_stop= rospy.Publisher("/parking_stop",         Bool,   queue_size=1, latch=True)

        # 상태 투표용 히스토리
        self.state_hist = collections.deque(maxlen=int(max(1000, self.history_sec*self.rate_hz*2)))
        self.last_decide_time = 0.0
        self.last_final_state = "unknown"

        # 서비스(Parallel/T 모드 확정)
        self.srv_t        = rospy.Service("/GetParkingPath", Trigger, self._srv_get_t)
        self.srv_parallel = rospy.Service("/GetParallelParkingPath", Trigger, self._srv_get_parallel)

        # YOLO 초기화 & 웹캠
        self._init_yolo_tl()
        self._init_yolo_cone()
        self.cap = None
        self._init_webcam()

        rospy.loginfo("[READY] webcam only | TL/Cone always-on | stop_topic(String)=%s", self.stop_topic)

    # ──────────────────────────────────────────────────
    # 공통/YOLO 초기화
    # ──────────────────────────────────────────────────
    def _common_imports(self):
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox
        return DetectMultiBackend, non_max_suppression, scale_boxes, letterbox

    def _init_yolo_tl(self):
        y5_root = Path(self.yolov5_dir).resolve()
        if not (y5_root/"models"/"common.py").exists():
            rospy.logerr("[YOLO-TL] wrong yolov5_dir: %s", str(y5_root))
            raise RuntimeError("yolov5_dir invalid")
        if str(y5_root) not in sys.path: sys.path.insert(0, str(y5_root))
        DetectMultiBackend, nms, scale_boxes, letterbox = self._common_imports()
        self.DetectMultiBackend = DetectMultiBackend
        self.nms = nms; self.scale_boxes = scale_boxes; self.letterbox = letterbox
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half   = (self.device.type != 'cpu')

        w_tl = Path(self.tl_weights)
        wpath = (Path(self.yolov5_dir)/w_tl) if not w_tl.is_absolute() else w_tl
        if not wpath.exists():
            alt = Path(self.tl_weights)
            if alt.exists(): wpath = alt
            else: raise FileNotFoundError(f"tl_weights not found: {self.tl_weights}")

        self.model = self.DetectMultiBackend(str(wpath), device=self.device, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        try: self.model.warmup(imgsz=(1,3,P("imgsz"),P("imgsz")))
        except Exception: pass

        self.tl_id_to_color: Dict[int, str] = {0: "red", 1: "green", 2: "yellow", 4: "rl"}
        self.tl_all_ids = set(self.tl_id_to_color.keys())
        rospy.loginfo("[YOLO-TL] fixed class-id map: %s", self.tl_id_to_color)

    def _init_yolo_cone(self):
        y5_root = Path(self.cone_y5_dir).resolve()
        if not (y5_root/"models"/"common.py").exists():
            rospy.logerr("[YOLO-CONE] wrong cone_yolov5_dir: %s", str(y5_root))
            raise RuntimeError("cone_yolov5_dir invalid")
        if str(y5_root) not in sys.path: sys.path.insert(0, str(y5_root))
        DetectMultiBackend, nms, scale_boxes, letterbox = self._common_imports()
        self.cone_nms = nms; self.cone_scale_boxes = scale_boxes; self.cone_letterbox = letterbox

        w_cone = Path(self.cone_weights)
        wpath = (Path(self.cone_y5_dir)/w_cone) if not w_cone.is_absolute() else w_cone
        if not wpath.exists():
            alt = Path(self.cone_weights)
            if alt.exists(): wpath = alt
            else: raise FileNotFoundError(f"cone_weights not found: {self.cone_weights}")

        self.cone_model = self.DetectMultiBackend(str(wpath), device=self.device, fp16=self.half)
        self.cone_stride, self.cone_names, self.cone_pt = self.cone_model.stride, self.cone_model.names, self.cone_model.pt
        try: self.cone_model.warmup(imgsz=(1,3,P("cone_imgsz"),P("cone_imgsz")))
        except Exception: pass

        if isinstance(self.cone_names, dict):
            mx = max(int(k) for k in self.cone_names.keys()); tmp = [""]*(mx+1)
            for k,v in self.cone_names.items(): tmp[int(k)] = v
            self.cone_names = tmp

        cone_label_list = _parse_labels(P("cone_label_csv"))
        name_to_ids: Dict[str, List[int]] = {}
        for i, n in enumerate(self.cone_names):
            if not n: continue
            name_to_ids.setdefault(n.strip().lower(), []).append(i)
        cone_ids = set()
        for alias in cone_label_list:
            if alias in name_to_ids: cone_ids.update(name_to_ids[alias]); continue
            for k,v in name_to_ids.items():
                if alias in k: cone_ids.update(v)
        self.cone_ids = cone_ids
        rospy.loginfo("[YOLO-CONE] ids=%s for labels=%s", sorted(self.cone_ids), cone_label_list)

    def _init_webcam(self):
        backend = cv2.CAP_DSHOW if os.name=='nt' else 0
        self.cap = cv2.VideoCapture(self.cam_index, backend)
        if self.cap_width  > 0: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_width)
        if self.cap_height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        if self.cap_fps    > 0: self.cap.set(cv2.CAP_PROP_FPS,          self.cap_fps)
        if not self.cap.isOpened(): raise RuntimeError("Webcam open failed (index=%d)" % self.cam_index)

    # 유틸: 빨간 텍스트(외곽선 포함)
    def _draw_text_red(self, img, text, org, scale=0.7, thickness=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,255), thickness, cv2.LINE_AA)

    # 루프
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                rospy.logwarn_throttle(2.0, "[Camera] empty frame"); rate.sleep(); continue
            self._process_and_publish(bgr)
            rate.sleep()

    # 메인 처리
    @torch.no_grad()
    def _process_and_publish(self, im0: np.ndarray):
        H, W = im0.shape[:2]
        now = self._now_sec()

        # TL 추론
        tl_state_instant = self._run_tl_yolo_color(im0, H, W)
        self._tl_append_state(now, tl_state_instant)
        self._tl_trim_history(now)

        # 상태 퍼블리시(주기적)
        publish_now = (now - self.last_decide_time) >= self.decide_period_sec
        if publish_now:
            self.last_final_state = self._tl_time_weighted_winner(now)
            self.last_decide_time = now
            self.pub_state.publish(String(data=self.last_final_state))
            self.pub_tlcolor.publish(String(data=self.last_final_state))

        # ── Cone: 좌/우 ROI (초기 파라미터 고정값 사용) ──
        left_roi  = self._compute_roi_from_ratios(self.cone_left_x0_ratio,  self.cone_left_x1_ratio,
                                                  self.cone_roi_top_ratio,  self.cone_roi_bottom_ratio, W, H)
        right_roi = self._compute_roi_from_ratios(self.cone_right_x0_ratio, self.cone_right_x1_ratio,
                                                  self.cone_roi_top_ratio,  self.cone_roi_bottom_ratio, W, H)
        hits, roi_rects = self._run_cone_inference_multi_roi(im0, H, W, {"left": left_roi, "right": right_roi})
        hit_left  = bool(hits.get("left", False))
        hit_right = bool(hits.get("right", False))
        hit_any   = hit_left or hit_right

        # 퍼블리시 + 서비스 집계(왼쪽=T, 오른쪽=평행)
        self.pub_cone_hit.publish(Bool(data=hit_any))
        self._hits_t.append((now, 1 if hit_left else 0))
        self._hits_parallel.append((now, 1 if hit_right else 0))

        # 디버그/GUI 오버레이
        if self.show_debug:
            try:
                if self.cone_draw_roi:
                    lx0, ly0, lx1, ly1 = roi_rects["left"]
                    rx0, ry0, rx1, ry1 = roi_rects["right"]
                    cv2.rectangle(im0, (lx0, ly0), (lx1, ly1), (255, 255, 0) if not hit_left else (0, 0, 255), 2 if not hit_left else 3)
                    cv2.rectangle(im0, (rx0, ry0), (rx1, ry1), (0, 255, 255) if not hit_right else (0, 0, 255), 2 if not hit_right else 3)

                tl_detected = (tl_state_instant is not None) and (tl_state_instant.lower() != "unknown")
                self._draw_text_red(im0, f"TL: {'DETECTED' if tl_detected else 'NONE'}  (PUB={self.last_final_state})",
                                    (10, 28), scale=0.8, thickness=2)
                self._draw_text_red(im0, f"CONE L: {'HIT' if hit_left else 'MISS'} | R: {'HIT' if hit_right else 'MISS'}",
                                    (10, 56), scale=0.8, thickness=2)
                # ROI 비율(초기값) 표기
                self._draw_text_red(
                    im0,
                    f"Lx[{self.cone_left_x0_ratio:.2f},{self.cone_left_x1_ratio:.2f}]  "
                    f"Rx[{self.cone_right_x0_ratio:.2f},{self.cone_right_x1_ratio:.2f}]  "
                    f"Y[{self.cone_roi_top_ratio:.2f},{self.cone_roi_bottom_ratio:.2f}]",
                    (10, 84), scale=0.7, thickness=2
                )
                if self.enable_gui:
                    cv2.imshow("TL + Cone (always-on)", im0)
                    cv2.waitKey(1)
            except Exception:
                pass

    # ──────────────────────────────────────────────────
    # 신호등 — YOLO (감지 박스 전부 그리기 + 대표 박스 강조)
    # ──────────────────────────────────────────────────
    def _run_tl_yolo_color(self, im0, H, W) -> str:
        im = self.letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2,0,1))[::-1]; im = np.ascontiguousarray(im)
        t  = torch.from_numpy(im).to(self.device); t = t.half() if self.half else t.float()
        t /= 255.0
        if t.ndimension()==3: t = t.unsqueeze(0)
        pred = self.model(t, augment=False)
        pred = self.nms(pred, self.conf_thres, self.iou_thres,
                        classes=list(self.tl_all_ids) if self.tl_all_ids else None,
                        max_det=self.max_det)

        cands = []
        draw_list = []
        cx, cy = W*0.5, H*0.5

        for det in pred:
            if not len(det): continue
            det_scaled = det.clone()
            det_scaled[:, :4] = self.scale_boxes(t.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det_scaled:
                conf = float(conf.item())
                if conf < self.tl_yolo_min_conf: continue
                cls_i = int(cls.item())
                color = self.tl_id_to_color.get(cls_i, None)
                if color is None: continue
                x1,y1,x2,y2 = [int(v.item()) for v in xyxy]
                x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W-1,x2), min(H-1,y2)
                if x2<=x1 or y2<=y1: continue
                w,h = (x2-x1),(y2-y1)
                if w<self.min_box_w or h<self.min_box_h: continue
                ar = w/float(h) if h>0 else 0.0
                if ar < 1.2: continue
                bx,by = 0.5*(x1+x2), 0.5*(y1+y2)
                dist2  = (bx-cx)**2 + (by-cy)**2
                area   = w*h
                cands.append((x1,y1,x2,y2, conf, color, dist2, area))
                draw_list.append((x1,y1,x2,y2, conf, color))

        if not cands:
            return "unknown"

        bx1,by1,bx2,by2, bconf, bcolor, _, _ = self._select_tl_box_yolo(cands)

        if self.show_debug:
            col_map = {"green": (0,255,0), "rl": (255,255,0), "yellow": (0,255,255), "red": (0,0,255)}
            if self.tl_draw_all_boxes:
                for x1,y1,x2,y2, conf, color in draw_list:
                    col = col_map.get(color, (200,200,200))
                    thick = self.tl_box_thickness
                    cv2.rectangle(im0, (x1,y1), (x2,y2), col, thick)
                    self._draw_text_red(im0, f"{color} {conf:.2f}", (x1, max(0,y1-8)), scale=0.7, thickness=2)
            col = col_map.get(bcolor, (200,200,200))
            cv2.rectangle(im0, (bx1,by1), (bx2,by2), col, self.tl_box_thickness+1)
            self._draw_text_red(im0, f"{bcolor} {bconf:.2f}", (bx1, max(0,by1-8)), scale=0.8, thickness=2)

        return bcolor

    def _select_tl_box_yolo(self, cands):
        m = self.selection_mode
        if m == "largest":        return max(cands, key=lambda c: c[7])
        if m == "highest_conf":   return max(cands, key=lambda c: c[4])
        if m == "leftmost":       return min(cands, key=lambda c: 0.5*(c[0]+c[2]))
        if m == "rightmost":      return max(cands, key=lambda c: 0.5*(c[0]+c[2]))
        if m == "center_closest": return min(cands, key=lambda c: c[6])
        if m == "hybrid":
            max_area  = max(x[7] for x in cands) or 1.0
            max_dist2 = max(x[6] for x in cands) or 1.0
            wc, wa = max(0.0, self.hybrid_center_weight), max(0.0, self.hybrid_area_weight)
            scored = []
            for x in cands:
                nd = x[6]/max_dist2; na = x[7]/max_area
                score = wc*nd - wa*na - 0.1*(1.0 - x[4])
                scored.append((score, x))
            return min(scored, key=lambda s: (s[0], -s[1][7]))[1]
        cands.sort(key=lambda c: (c[6], -c[7], -c[4]))
        return cands[0]

    # TL 투표·유틸
    def _now_sec(self) -> float:
        t = rospy.get_time()
        return t if t > 0 else time.time()

    def _tl_append_state(self, t_sec: float, state: str):
        self.state_hist.append((t_sec, state))

    def _tl_trim_history(self, now_sec: float):
        cutoff = now_sec - max(self.history_sec, self.vote_time_window_sec * 3.0)
        while self.state_hist and self.state_hist[0][0] < cutoff:
            self.state_hist.popleft()

    def _tl_time_weighted_winner(self, now_sec: float) -> str:
        if not self.state_hist: return "unknown"
        window = self.vote_time_window_sec
        cutoff = now_sec - window
        events = list(self.state_hist)
        idx = len(events) - 1
        while idx >= 0 and events[idx][0] > cutoff: idx -= 1
        state_at_cut = events[idx][1] if idx >= 0 else events[0][1]
        segs = [(cutoff, state_at_cut)]
        for t, s in events:
            if cutoff <= t <= now_sec: segs.append((t, s))
        last_state = segs[-1][1] if segs else events[-1][1]
        segs.append((now_sec, last_state))
        acc = {"green": 0.0, "rl": 0.0, "red": 0.0, "yellow": 0.0, "unknown": 0.0}
        for i in range(len(segs)-1):
            t0, s0 = segs[i]; t1, _ = segs[i+1]
            acc[s0 if s0 in acc else "unknown"] += max(0.0, t1 - t0)
        prio = {"green": 4, "rl": 3, "red": 2, "yellow": 1, "unknown": 0}
        return max(acc.items(), key=lambda kv: (kv[1], prio.get(kv[0], -1)))[0]

    # ──────────────────────────────────────────────────
    # 라바콘 ROI/추론 (멀티 ROI: 좌/우 동시 판정)
    # ──────────────────────────────────────────────────
    def _compute_roi_from_ratios(self, x0r: float, x1r: float, y0r: float, y1r: float, W: int, H: int):
        x0r = max(0.0, min(1.0, x0r)); x1r = max(0.0, min(1.0, x1r))
        y0r = max(0.0, min(1.0, y0r)); y1r = max(0.0, min(1.0, y1r))
        x0 = int(round(x0r * W)); x1 = int(round(x1r * W))
        y0 = int(round(y0r * H)); y1 = int(round(y1r * H))
        if x1 <= x0: x1 = min(W, x0 + 1)
        if y1 <= y0: y1 = min(H, y0 + 1)
        x0 = max(0, min(W-1, x0)); x1 = max(1, min(W, x1))
        y0 = max(0, min(H-1, y0)); y1 = max(1, min(H, y1))
        return (x0, y0, x1, y1)

    @torch.no_grad()
    def _run_cone_inference_multi_roi(self, im0, H, W, roi_rects: Dict[str, Tuple[int,int,int,int]]):
        im = self.cone_letterbox(im0, self.cone_imgsz, stride=self.cone_stride, auto=self.cone_pt)[0]
        im = im.transpose((2,0,1))[::-1]; im = np.ascontiguousarray(im)
        t  = torch.from_numpy(im).to(self.device); t = t.half() if self.half else t.float()
        t /= 255.0
        if t.ndimension()==3: t = t.unsqueeze(0)
        pred = self.cone_model(t, augment=False)
        pred = self.cone_nms(pred, self.cone_conf_thres, self.cone_iou_thres,
                             classes=None, max_det=self.cone_max_det)

        hits = {k: False for k in roi_rects.keys()}
        def all_hit(): return all(hits.values())

        for det in pred:
            if not len(det): continue
            det_scaled = det.clone()
            det_scaled[:, :4] = self.cone_scale_boxes(t.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det_scaled:
                cls_i = int(cls.item())
                if self.cone_ids and (cls_i not in self.cone_ids): continue
                x1,y1,x2,y2 = [int(v.item()) for v in xyxy]
                x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W-1,x2), min(H-1,y2)
                if x2<=x1 or y2<=y1: continue
                area_ratio = ((x2-x1)*(y2-y1)) / float(W*H)
                if area_ratio < self.cone_min_box_area_ratio: continue

                for k,(rx0,ry0,rx1,ry1) in roi_rects.items():
                    ix1, iy1 = max(x1, rx0), max(y1, ry0)
                    ix2, iy2 = min(x2, rx1), min(y2, ry1)
                    if (ix2 > ix1) and (iy2 > iy1):
                        hits[k] = True
                if all_hit(): break
            if all_hit(): break

        return hits, roi_rects

    # 서비스(웨이포인트 게이팅 제거)
    def _srv_get_t(self, _req):
        return self._finalize_parking_mode("t")
    def _srv_get_parallel(self, _req):
        return self._finalize_parking_mode("parallel")
    def _finalize_parking_mode(self, mode: str) -> TriggerResponse:
        start = self._now_sec()
        hits_deque = self._hits_t if mode == "t" else self._hits_parallel
        while hits_deque and hits_deque[0][0] < start:
            hits_deque.popleft()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            now = self._now_sec()
            if (now - start) >= self.cone_window_sec: break
            rate.sleep()
        now = self._now_sec()
        hits = sum(h for (t,h) in list(hits_deque) if start <= t <= now)
        ok = (hits >= self.cone_min_hits)
        if ok:
            if mode == "t": self.pub_t.publish(String(data="2"))
            else:           self.pub_parallel.publish(String(data="1"))
            self.pub_parking_stop.publish(Bool(data=True))
            code = "2" if mode == "t" else "1"
            return TriggerResponse(success=True, message=f"confirmed {mode}, hits={hits}/{self.cone_min_hits}, code={code}")
        else:
            elapsed = now - start
            return TriggerResponse(success=False, message=f"insufficient hits {hits}/{self.cone_min_hits} in {elapsed:.2f}s")

# ─────────────────────────────────────────────────────
def main():
    rospy.init_node("tl_cone_simple_node", anonymous=False)
    node = None
    try:
        node = TLAndConeNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            if node and getattr(node, "cap", None): node.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()

