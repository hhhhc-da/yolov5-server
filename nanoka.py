#coding=utf-8
# Ultralytics YOLOv5, AGPL-3.0 license
from utils.general import cv2, non_max_suppression
from models.common import DetectMultiBackend
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import threading
import io
import cv2

# ----------------------------------------------------------------------------------------------------------------------
# Model parameters
imgsz = (640, 640)
stat = ['apple', 'orange']
# Cache
yolo_cache = np.zeros([640, 640, 3])
# YOLOv5 model handler
model = DetectMultiBackend('yolov5s.pt', data='data/coco128.yaml') # 运行在 CPU 上
# Load camera
cap = cv2.VideoCapture(0)
# Flask
app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------------------------------------------------------
# 绘制图像函数 (坐标使用 xyxy 方式)
def plot_xyxy(x, img, color=(0, 0, 255), label=None, line_thickness=3):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness,
                  lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(
            label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)


# ----------------------------------------------------------------------------------------------------------------------
# 测试用 API, 直接调取就可以实时显示, 前提要先打开摄像头
def val_vedio():
    global cap
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, imgsz)

        # NHWC to NCHW
        image = frame.copy()
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image) / 255.0

        # Inference
        pred = model(torch.from_numpy(image).unsqueeze(0).float())

        # NMS algorithm
        pred = non_max_suppression(pred)

        # Process predictions
        for i, det in enumerate(pred):
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    types = ['person', 'cat']
                    if model.names[c] not in types:
                        continue

                    label = '{} {:4f}'.format(
                        model.names[c], float(conf.item()))

                    plot_xyxy(xyxy, frame, label=label,
                              color=(0, 0, 255), line_thickness=2)

                    # print("xyxy:", [float(xyxy[i].item())
                    #       for i in range(4)], "\tconf:", float(conf.item()))

            # Stream results
            cv2.imshow('Person', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Done.")
                return


# ----------------------------------------------------------------------------------------------------------------------
# 推理函数
def inference(frame):
    global cap, yolo_cache

    # 构造统计信息
    statistic = {}
    for i, k in enumerate(stat):
        statistic[k] = 0

    # 原图片记录
    shape = yolo_cache.shape

    # NHWC to NCHW
    yolo_cache = cv2.resize(frame, imgsz)
    image = yolo_cache.transpose(2, 0, 1)
    image = np.ascontiguousarray(image) / 255.0

    # Inference
    pred = model(torch.from_numpy(image).unsqueeze(0).float())

    # NMS algorithm
    pred = non_max_suppression(pred)

    # Process predictions
    for i, det in enumerate(pred):
        if len(det):
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class

                types = list(statistic.keys())
                if model.names[c] not in types:
                    continue

                plot_xyxy(xyxy, yolo_cache, label='{:4f}'.format(float(conf.item())),color=(0, 0, 255), line_thickness=2)
                statistic[model.names[c]] = 1

    yolo_cache = cv2.resize(yolo_cache, (shape[0],shape[1])) # 变换回原来的尺寸
    return statistic

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def root():
    return render_template('50x.html'), 500

# 图片传输函数, 用于传输测试图片
@app.route('/image', methods=['GET'])
def image():
    global yolo_cache

    name = {}
    for i, k in enumerate(stat):
        name[i] = k

    _, frame = cap.read()

    statistic = inference(frame)
    result = list(statistic.values())
    if sum(result) > 1:
        print("出现了至少两个类别")
    elif sum(result) == 0:
        print("没有检测到任何物体")
    else:
        print("类别为 {}".format(name[result.index(1)]))

    _, img_encoded = cv2.imencode('.jpg', yolo_cache)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# 图片类别请求函数, -1 表示物体数量过多, -2 表示没有物体, 其他都是物体标签
@app.route('/request', methods=['GET'])
def reqfc():
    _, frame = cap.read()

    statistic = inference(frame)
    result = list(statistic.values())

    if sum(result) > 1:
        return jsonify({'code': -1, 'info': -1}), 200
    elif sum(result) == 0:
        return jsonify({'code': -1, 'info': -2}), 200
    else:
        return jsonify({'code': 0, 'info': result.index(1)}), 200


@app.route('/val', methods=['GET'])
def val():
    thread1 = threading.Thread(target=val_vedio)
    thread1.start()
    return jsonify({'code': 0, 'info': 'Success'}), 200

# ----------------------------------------------------------------------------------------------------------------------

try:
    # Run Flask serve
    app.run('0.0.0.0', port=5000)
except Exception as e:
    print("捕捉到错误:", e)
finally:
    cap.release()

