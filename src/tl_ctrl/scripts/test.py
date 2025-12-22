#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, pathlib
import cv2, torch
import numpy as np
from pathlib import Path

# ⚠️ WindowsPath → PosixPath 매핑 (윈도우에서 저장된 pt도 로드 가능)
try:
    pathlib.WindowsPath = pathlib.PosixPath
except Exception:
    pass

# YOLOv5 소스 경로 (수정 가능)
YOLO_DIR = f"/home/{os.getenv('USER')}/catkin_ws/src/yolov5"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# 가중치 파일 경로
WEIGHTS = "/home/icas/catkin_ws/src/tl_ctrl/scripts/weights/tl/best.pt"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(WEIGHTS, device=device, fp16=(device.type != "cpu"))
    stride, names, pt = model.stride, model.names, model.pt

    print(f"[INFO] 모델 클래스: {names}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 카메라 프레임 읽기 실패")
            break

        # 전처리
        img = letterbox(frame, 640, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # 추론
        pred = model(im, augment=False)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=100)

        # 결과 처리
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("YOLOv5 Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
