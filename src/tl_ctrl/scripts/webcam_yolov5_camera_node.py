#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
웹캠 영상만으로 장애물 탐지(Depth 없음).
- 신호등: 좌/우 2분할 고정판정(오른쪽=green, 왼쪽=red). 동시 검출 시 green 우선.
- 장애물: COCO 계열 객체(person, bicycle, car, motorcycle, bus, truck 등) 검출.
- 위험판정(Depth 대체): 화면 하단 근접 + 바운딩박스 크기 기준 휴리스틱.
  · 조건: bbox 중심 y > h*risk_band_y  AND  (hfrac > risk_h_frac  OR  areafrac > risk_area_frac)
퍼블리시:
- /traffic_light/state : std_msgs/String  ["green","red","unknown"]
- /traffic_light/stop  : std_msgs/Bool    [red→True]
- /obstacle/alert      : std_msgs/Bool    [위험판정시 True]
- /obstacle/info       : std_msgs/String  [JSON: {boxes:[{cls,conf,x1,y1,x2,y2,risky}], fps}]

ROS Param 예:
- ~image_topic: "/camera/color/image_raw"
- ~weights: YOLOv5 TL 전용 가중치(.pt) 경로 또는 빈 값(빈 값이면 yolov5s로 대체 로드)
- ~obs_weights: 장애물용 가중치(.pt). 기본: yolov5s(pretrained)
- ~split_x_ratio: 0.5
- ~conf_thres / ~iou_thres / ~imgsz (신호등)
- ~obs_conf / ~obs_iou / ~obs_imgsz (장애물)
- ~obs_targets: "person,bicycle,car,motorcycle,bus,truck"
- 위험 휴리스틱: ~risk_band_y(0.55) ~risk_h_frac(0.35) ~risk_area_frac(0.10)
"""

import os
import sys
import time
import json
import collections
from collections import deque
from typing import List, Tuple

import numpy as np
import cv2
import torch

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge


# ---------- 공용 전처리 ----------
def letterbox(im, new_shape=(640, 640), stride=32, auto=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)

def scale_boxes(img_shape, boxes, orig_shape, ratio_pad=None):
    if ratio_pad:
        r, (dw, dh) = ratio_pad
    else:
        r = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
        dw, dh = (img_shape[1] - orig_shape[1] * r) / 2, (img_shape[0] - orig_shape[0] * r) / 2
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= r
    boxes[:, 0].clip(0, orig_shape[1], out=boxes[:, 0])
    boxes[:, 1].clip(0, orig_shape[0], out=boxes[:, 1])
    boxes[:, 2].clip(0, orig_shape[1], out=boxes[:, 2])
    boxes[:, 3].clip(0, orig_shape[0], out=boxes[:, 3])
    return boxes

def nms(boxes, scores, iou_thres=0.45, max_det=300):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# ---------- 노드 ----------
class TrafficLightNode:
    def __init__(self):
        # 토픽
        self.image_topic   = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.status_topic  = rospy.get_param("~status_topic", "/traffic_light/state")
        self.stop_topic    = rospy.get_param("~stop_topic",   "/traffic_light/stop")
        self.obs_alert_topic = rospy.get_param("~obs_alert_topic", "/obstacle/alert")
        self.obs_info_topic  = rospy.get_param("~obs_info_topic",  "/obstacle/info")
        self.show_debug    = bool(rospy.get_param("~show_debug", False))

        # 신호등 분할
        self.split_x_ratio = float(rospy.get_param("~split_x_ratio", 0.5))

        # 신호등 모델
        self.weights       = rospy.get_param("~weights", "")
        self.imgsz         = int(rospy.get_param("~imgsz", 640))
        self.conf_thres    = float(rospy.get_param("~conf_thres", 0.25))
        self.iou_thres     = float(rospy.get_param("~iou_thres", 0.45))
        self.max_det       = int(rospy.get_param("~max_det", 100))

        # 장애물 모델
        self.obs_weights   = rospy.get_param("~obs_weights", "")
        self.obs_imgsz     = int(rospy.get_param("~obs_imgsz", 640))
        self.obs_conf      = float(rospy.get_param("~obs_conf", 0.35))
        self.obs_iou       = float(rospy.get_param("~obs_iou", 0.45))
        self.obs_max_det   = int(rospy.get_param("~obs_max_det", 100))
        self.obs_targets_s = rospy.get_param("~obs_targets", "person,bicycle,car,motorcycle,bus,truck")

        # 위험 휴리스틱(Depth 대체)
        self.risk_band_y     = float(rospy.get_param("~risk_band_y", 0.55))  # 하단 밴드 시작 비율
        self.risk_h_frac     = float(rospy.get_param("~risk_h_frac", 0.35))  # 화면 높이 대비 bbox 높이
        self.risk_area_frac  = float(rospy.get_param("~risk_area_frac", 0.10)) # 화면 면적 대비 bbox 면적

        # 디바이스
        self.device        = rospy.get_param("~device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.half          = bool(rospy.get_param("~half", True)) and ("cuda" in self.device)

        # 상태
        self.hist_size     = int(rospy.get_param("~hist_size", 5))
        self.state_hist    = deque(maxlen=self.hist_size)

        # ROS IO
        self.bridge        = CvBridge()
        self.pub_state     = rospy.Publisher(self.status_topic, String, queue_size=10)
        self.pub_stop      = rospy.Publisher(self.stop_topic,   Bool,   queue_size=10)
        self.pub_obs_alert = rospy.Publisher(self.obs_alert_topic, Bool, queue_size=10)
        self.pub_obs_info  = rospy.Publisher(self.obs_info_topic,  String, queue_size=10)
        self.sub_img       = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)

        # 모델 로드
        self.model_tl, self.names_tl, self.stride_tl = self.load_model(self.weights, fallback="yolov5s")
        self.model_obs, self.names_obs, self.stride_obs = self.load_model(self.obs_weights, fallback="yolov5s")

        # 장애물 타깃 클래스 ID 계산
        target_names = [s.strip() for s in self.obs_targets_s.split(",") if s.strip()]
        self.obs_target_ids = set([i for i, n in enumerate(self.names_obs) if n in target_names])

        rospy.loginfo(f"[TL] subscribe: {self.image_topic}")
        rospy.loginfo(f"[TL] publish: {self.status_topic}, {self.stop_topic}, {self.obs_alert_topic}, {self.obs_info_topic}")
        rospy.loginfo(f"[TL] device={self.device}, TL_imgsz={self.imgsz}, OBS_imgsz={self.obs_imgsz}, split={self.split_x_ratio}")
        rospy.loginfo(f"[OBS] targets={sorted(list(self.obs_target_ids))} ({target_names})")

        self.t_prev = time.time()
        self.fps = 0.0

    def load_model(self, weights_path: str, fallback: str = "yolov5s"):
        model = None
        names = ['obj']
        stride = 32
        if weights_path and os.path.isfile(weights_path):
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, source='local')
                names = model.names if hasattr(model, 'names') else names
                stride = int(getattr(model, 'stride', [32])[0]) if hasattr(model, 'stride') else 32
                rospy.loginfo(f"[LOAD] local weights: {weights_path}")
            except Exception as e:
                rospy.logwarn(f"[LOAD] local load failed: {e}")
        if model is None:
            try:
                model = torch.hub.load('ultralytics/yolov5', fallback, pretrained=True)
                names = model.names if hasattr(model, 'names') else names
                stride = int(getattr(model, 'stride', [32])[0]) if hasattr(model, 'stride') else 32
                rospy.loginfo(f"[LOAD] fallback model: {fallback}")
            except Exception as e:
                rospy.logerr(f"[LOAD] failed: {e}")
                raise
        model.to(self.device)
        if self.half and hasattr(model, 'model'):
            model.model.half()
        model.eval()
        return model, names, stride

    @torch.inference_mode()
    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[cv_bridge] {e}")
            return

        t0 = time.time()

        # 신호등
        tl_decisions, vis = self.run_tl(bgr)
        tl_state = self.majority_vote(tl_decisions)
        self.pub_state.publish(String(data=tl_state))
        self.pub_stop.publish(Bool(data=(tl_state == "red")))

        # 장애물
        obs, alert = self.run_obstacle(bgr, vis if self.show_debug else None)
        self.pub_obs_alert.publish(Bool(data=alert))
        self.pub_obs_info.publish(String(data=json.dumps(obs, ensure_ascii=False)))

        # 디버그
        if self.show_debug:
            try:
                h, w = bgr.shape[:2]
                split_x = int(w * self.split_x_ratio)
                cv2.line(vis, (split_x, 0), (split_x, h), (255, 255, 255), 1)
                if tl_state != "unknown":
                    cv2.putText(vis, f"TL: {tl_state.upper()}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0) if tl_state == "green" else (0, 0, 255), 2, cv2.LINE_AA)
                color = (0, 0, 255) if alert else (0, 255, 0)
                cv2.putText(vis, f"OBS_ALERT: {bool(alert)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.putText(vis, f"FPS: {self.fps:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("YOLOv5 | TL+OBS (webcam)", vis)
                cv2.waitKey(1)
            except Exception:
                pass

        # FPS
        dt = max(1e-6, time.time() - self.t_prev)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.t_prev = time.time()

    # ---------- 신호등: 좌/우 고정판정 ----------
    def run_tl(self, bgr):
        im0 = bgr.copy()
        im, r, (dw, dh) = letterbox(im0, self.imgsz, stride=self.stride_tl, auto=True)
        im = im[:, :, ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(im)
        im_t = torch.from_numpy(im).to(self.device)
        im_t = im_t.half() if self.half else im_t.float()
        im_t /= 255.0
        if im_t.ndimension() == 3:
            im_t = im_t.unsqueeze(0)

        pred = self.model_tl(im_t, augment=False, visualize=False)[0]
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()

        det = pred[pred[:, 4] >= self.conf_thres] if pred.size else np.empty((0,6), np.float32)
        if det.shape[0]:
            keep = nms(det[:, :4], det[:, 4], iou_thres=self.iou_thres, max_det=self.max_det)
            det = det[keep]
        else:
            det = np.empty((0,6), np.float32)

        decisions = []
        h0, w0 = im0.shape[:2]
        split_x = int(w0 * self.split_x_ratio)

        if det.shape[0]:
            boxes = det[:, :4].copy()
            boxes = scale_boxes((self.imgsz, self.imgsz), boxes, im0.shape, ratio_pad=(r, (dw, dh))).round()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                conf = float(det[i, 4])
                cx = 0.5 * (x1 + x2)
                state = "green" if cx >= split_x else "red"
                decisions.append((state, conf, (x1, y1, x2, y2)))
                if self.show_debug:
                    color = (0,255,0) if state == "green" else (0,0,255)
                    cv2.rectangle(im0, (x1,y1), (x2,y2), color, 2)
        return decisions, im0

    def majority_vote(self, decisions):
        states_now = [d[0] for d in decisions]
        if "green" in states_now:
            current = "green"
        elif "red" in states_now:
            current = "red"
        else:
            current = "unknown"
        self.state_hist.append(current)
        counts = collections.Counter(self.state_hist)
        prio = {"green":2, "red":1, "unknown":0}
        winner = max(counts.items(), key=lambda kv: (kv[1], prio.get(kv[0], -1)))[0]
        return winner

    # ---------- 장애물: COCO 검출 + 위험 휴리스틱 ----------
    def run_obstacle(self, bgr, vis=None):
        H, W = bgr.shape[:2]
        im0 = bgr if vis is None else vis  # 디버그 표시 시 동일 버퍼 사용

        im, r, (dw, dh) = letterbox(bgr, self.obs_imgsz, stride=self.stride_obs, auto=True)
        im = im[:, :, ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(im)
        im_t = torch.from_numpy(im).to(self.device)
        im_t = im_t.half() if self.half else im_t.float()
        im_t /= 255.0
        if im_t.ndimension() == 3:
            im_t = im_t.unsqueeze(0)

        pred = self.model_obs(im_t, augment=False, visualize=False)[0]
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()

        det = pred[pred[:, 4] >= self.obs_conf] if pred.size else np.empty((0,6), np.float32)
        if det.shape[0]:
            keep = nms(det[:, :4], det[:, 4], iou_thres=self.obs_iou, max_det=self.obs_max_det)
            det = det[keep]
        else:
            det = np.empty((0,6), np.float32)

        boxes_json = []
        alert = False
        y_band = H * self.risk_band_y

        if det.shape[0]:
            boxes = det[:, :4].copy()
            boxes = scale_boxes((self.obs_imgsz, self.obs_imgsz), boxes, (H, W), ratio_pad=(r, (dw, dh))).round()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                conf = float(det[i, 4])
                cls  = int(det[i, 5]) if det.shape[1] > 5 else 0

                if cls not in self.obs_target_ids:
                    continue

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                cx, cy = x1 + w * 0.5, y1 + h * 0.5
                hfrac = h / float(H)
                areaf = (w * h) / float(W * H)

                risky = (cy >= y_band) and (hfrac >= self.risk_h_frac or areaf >= self.risk_area_frac)
                alert = alert or risky

                name = self.names_obs[cls] if cls < len(self.names_obs) else str(cls)
                boxes_json.append({
                    "cls": name, "conf": round(conf, 3),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "risky": bool(risky)
                })

                if self.show_debug and im0 is not None:
                    col = (0,0,255) if risky else (0,255,255)
                    cv2.rectangle(im0, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(im0, f"{name} {conf:.2f}{' R' if risky else ''}",
                                (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        if self.show_debug and im0 is not None:
            yb = int(y_band)
            cv2.line(im0, (0, yb), (W, yb), (200, 200, 200), 1)

        payload = {"boxes": boxes_json, "fps": round(self.fps, 2)}
        return payload, bool(alert)


def main():
    rospy.init_node("l515_yolov5_traffic_light_node", anonymous=False)
    node = None
    try:
        node = TrafficLightNode()
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
