import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from ultralytics import RTDETR
import cv2

css_style = """
.c1 textarea {
    font-size: 20px !important;
    color: darkgray !important;
    font-weight: bold !important;
}
"""

coco_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
             'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
             'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
voc_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
model1 = YOLO('./ultralytics_new/yolov5su.pt')
model2 = YOLO('./ultralytics_new/runs/detect/train9/weights/best.pt')
model3 = YOLO('./ultralytics_new/yolov5mu.pt')
model4 = YOLO('./ultralytics_new/yolov8s.pt')
model5 = YOLO('./ultralytics_new/yolov8m.pt')
model6 = YOLO('./ultralytics_new/runs/detect/train3/weights/best.pt')
model7 = YOLO('./ultralytics_new/runs/detect/train6/weights/best.pt')
model8 = YOLO('./ultralytics_new/runs/detect/train7/weights/best.pt')
model9 = YOLO('./ultralytics_new/runs/detect/train8/weights/best.pt')
model10 = YOLO('./ultralytics_new/yolov9c.pt')
model11 = YOLO('./ultralytics_new/yolov9e.pt')
model12 = YOLO('./ultralytics_new/yolov8m-worldv2.pt')
model13 = YOLO('./ultralytics_new/yolov8l-worldv2.pt')
model14 = YOLO('./ultralytics_new/runs/detect/train10/weights/best.pt')
model15 = RTDETR('./ultralytics_new/rtdetr-l.pt')
model16 = RTDETR('./ultralytics_new/rtdetr-x.pt')
model17 = YOLO('./ultralytics_new/runs/detect/train11/weights/best.pt')
model18 = YOLO('./ultralytics_new/runs/detect/train12/weights/best.pt')
model19 = YOLO('./ultralytics_new/runs/detect/train18/weights/best.pt')
model20 = YOLO('./ultralytics_new/runs/detect/train19/weights/best.pt')
model21 = YOLO('./ultralytics_new/runs/detect/train17/weights/best.pt')

# title = "基于Gradio的YOLO演示项目"
# desc = "这是一个基于Gradio的YOLO演示项目，可选择多个yolo模型，多种训练的数据集，支持图片、视频、摄像头上传等！"
base_conf, base_iou = 0.25, 0.45

def predict_video_v51(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov5s进行预测
            results = model1(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v52(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov5train进行预测
            results = model2(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v53(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov5m进行预测
            results = model3(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v81(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8s进行预测
            results = model4(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v82(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8m进行预测
            results = model5(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v83(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8spt_coco128_100epochs进行预测
            results = model6(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v84(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8s_coco128_500epochs进行预测
            results = model7(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v85(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8scbam_coco128_500epochs进行预测
            results = model8(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v86(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8sema_coco128_500epochs进行预测
            results = model9(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v87(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8sta_coco128_500epochs进行预测
            results = model18(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v88(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8sptcbam_voc_30epochs进行预测
            results = model19(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v89(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8sptta_voc_30epochs进行预测
            results = model20(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v810(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用v8sptema_voc_30epochs进行预测
            results = model21(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v91(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov9c进行预测
            results = model10(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v92(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov9e进行预测
            results = model11(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v93(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov9train进行预测
            results = model14(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_v94(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov9train进行预测
            results = model17(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_w1(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8m-world进行预测
            results = model12(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_w2(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8l-world进行预测
            results = model13(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_rt1(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用rtdetr-l进行预测
            results = model15(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def predict_video_rt2(video, conf_thres, iou_thres, choices):
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultralytics_new/data/video/output.mp4', fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用rtdetr-x进行预测
            results = model16(frame, conf=conf_thres, iou=iou_thres, classes=choices)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    return 'ultralytics_new/data/video/output.mp4'

def horizontal_flip1(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip1(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def crop_image1(image, left, right, top, bottom, contrast, brightness):
    width, height = image.size
    left = int(left * width)
    right = int(right * width)
    top = int(top * height)
    bottom = int(bottom * height)
    image1 = image.crop((left, top, right, bottom))
    # 对比度调整
    enhancer = ImageEnhance.Contrast(image1)
    image2 = enhancer.enhance(contrast)
    # 亮度调整
    enhancer = ImageEnhance.Brightness(image2)
    image3 = enhancer.enhance(brightness)
    return image3

def create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness):
    if horizontal_flip and vertical_flip:
        img_new = crop_image1(vertical_flip1(horizontal_flip1(img)), left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    elif horizontal_flip:
        img_new = crop_image1(horizontal_flip1(img), left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    elif vertical_flip:
        img_new = crop_image1(vertical_flip1(img), left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    else:
        img_new = crop_image1(img, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    return img_new

def print_ans(result):
    ans = "识别到图片中物体及置信度如下：\n"
    count = 1
    for i in range(len(result.boxes.cls)):
        ans = ans + result.names[int(result.boxes.cls[i])] + str(count) + ":conf=" + str(
            round(float(result.boxes.conf[i]), 3)) + "    "
        count += 1
        if i + 1 < len(result.boxes.cls) and result.boxes.cls[i + 1] != result.boxes.cls[i]:
            count = 1
            ans = ans + "\n"
    return ans

def predict_image1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model4.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image2(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model5.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image3(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model6.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image4(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model7.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image5(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model8.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image6(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model9.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image7(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model10.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image8(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model11.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image9(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model12.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image10(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model13.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image11(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model14.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image12(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model15.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image13(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model16.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image14(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model17.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image15(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model18.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image16(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model19.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image17(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model20.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image18(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model21.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image51(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model1.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image52(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model2.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_image53(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres):
    img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness)
    results = model3.predict(source=img_new, conf=conf_thres, iou=iou_thres)
    im_array = results[0].plot()
    ans = print_ans(results[0])
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img, ans

def predict_directory51(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model1.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory52(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model2.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory53(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model3.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory81(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model4.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory82(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model5.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory83(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model6.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory84(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model7.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory85(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model8.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory86(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model9.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory87(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model18.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory88(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model19.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory89(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model20.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory810(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model21.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory91(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model10.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory92(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model11.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory93(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model14.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directory94(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model17.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directoryw1(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model12.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directoryw2(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model13.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directoryrt1(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model15.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

def predict_directoryrt2(directory, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop, contrast, brightness, conf_thres, iou_thres, choices):
    output = []
    for item in directory:
        img = Image.open(item)
        img_new = create_img1(img, horizontal_flip, vertical_flip, left_crop, right_crop, top_crop, bottom_crop,
                              contrast, brightness)
        results = model16.predict(source=img_new, conf=conf_thres, iou=iou_thres, classes=choices)
        im_array = results[0].plot()
        pil_img = Image.fromarray(im_array[..., ::-1])
        output.append(pil_img)
    return output

image_url1 = "https://image.cha138.com/20210706/25a32c48c181400ba8ca2b09f003935e.jpg"
image_url2 = "https://www.logo9.net/userfiles/images/9XIAMENUN.jpg"
image_url3 = "https://img2.baidu.com/it/u=1572920788,4071073152&fm=253&fmt=auto&app=138&f=PNG?w=1187&h=500"
image_url4 = "https://s3.bmp.ovh/imgs/2024/04/07/e2a0d2c093a10a44.png"
image_url5 = "https://s3.bmp.ovh/imgs/2024/03/26/6c7a3ca9fd470293.png"
image_url6 = "https://s3.bmp.ovh/imgs/2024/03/31/1c7565030eb7db22.png"
image_url7 = "https://s3.bmp.ovh/imgs/2024/04/03/fdcec33105ebe280.jpg"
image_url8 = "https://s3.bmp.ovh/imgs/2024/04/03/d56eaf6a1f087301.jpg"
image_url9 = "https://s3.bmp.ovh/imgs/2024/04/03/df2622905907878b.jpg"
image_url10 = "https://s3.bmp.ovh/imgs/2024/04/03/3ea181d9332fa03a.jpg"
image_url11 = "https://s3.bmp.ovh/imgs/2024/04/03/a9658ec14efef418.jpg"
image_url12 = "https://s3.bmp.ovh/imgs/2024/04/07/171118d64b785ac0.png"
image_url13 = "https://s3.bmp.ovh/imgs/2024/04/17/4ea59db15d91d3cd.jpg"
image_url14 = "https://s3.bmp.ovh/imgs/2024/04/18/3a4f188e25518b41.jpg"
image_url15 = "https://s3.bmp.ovh/imgs/2024/04/18/fa325d5e22d61a7f.jpg"
image_url16 = "https://s3.bmp.ovh/imgs/2024/04/25/f4d0c3c3a185adc3.jpg"
image_url17 = "https://s3.bmp.ovh/imgs/2024/04/25/209c934e26e19d47.jpg"
image_url18 = "https://s3.bmp.ovh/imgs/2024/04/25/f0b1ebe31c6741c8.jpg"

html_content = f"<h1 style='float: left; margin-left: 50px; margin-top: 40px'>基于Gradio的目标检测可视化平台</h1>" \
               f"<img src='{image_url1}' alt='' width='216' style='float: left; margin-left: 210px;'>" \
               f"<img src='{image_url2}' alt='' width='192' style='float:right;'>" \
               f"<div style='clear: both;'></div>" \
               f"<div style='margin-left: 30px; font-size: large; font-family: 仿宋'>这是一个基于Gradio的目标检测可视化平台，可选择多个目标检测主流前沿模型，支持多种训练的数据集，支持图片、文件夹、视频、摄像头上传等！</div>"
html_content2 = f"<img src='{image_url3}' alt='' width='650' style='margin-left: 30px; margin-top: 50px; margin-bottom: 50px'>" \
                f"<img src='{image_url5}' alt='' width='650' style='margin-left: 20px; margin-top: 50px; margin-bottom: 50px'>"
html_content3 = f"<img src='{image_url4}' alt='' width='900' style='margin-left: 250px; margin-top: 20px; margin-bottom: 20px'>"
html_content4 = f"<img src='{image_url6}' alt='' width='900' style='margin-left: 250px; margin-top: 20px; margin-bottom: 20px'>"
html_content5 = f"<h3 style='text-align: center; font-family: 仿宋; color: #00A8FF'>更多最新Yolov8内容，请点击下方官方文档查看</h3>" \
                f"<a href='https://docs.ultralytics.com/' style='display: block; text-align: center'>Ultralytics v8.1.0</a>"
html_content6 = f"<img src='{image_url7}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content7 = f"<img src='{image_url8}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content8 = f"<div style='text-align: center; font-family: 仿宋; font-size: large; font-weight: bold; color: black; margin-top: 20px; margin-bottom: 20px'>论文参考：" \
                f"<a href='https://arxiv.org/abs/1807.06521'>CBAM:Convolutional Block Attention Module</a></div>" \
                f"<img src='{image_url9}' alt='' width='700' style='margin-left: 350px; margin-bottom: 20px'>"
html_content9 = f"<img src='{image_url10}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content10 = f"<div style='text-align: center; font-family: 仿宋; font-size: large; font-weight: bold; color: black; margin-top: 20px; margin-bottom: 20px'>论文参考：" \
                f"<a href='https://arxiv.org/abs/2305.13563v1'>Efficient Multi-Scale Attention Module with Cross-Spatial Learning</a></div>" \
                f"<img src='{image_url11}' alt='' width='700' style='margin-left: 350px; margin-bottom: 20px'>"
html_content11 = f"<h3 style='text-align: center; font-family: 仿宋; color: #ea5470'>更多最新Yolov9内容，请点击下方官方文档查看</h3>" \
                f"<a href='https://docs.ultralytics.com/models/yolov9/' style='display: block; text-align: center'>YOLOv9: A Leap Forward in Object Detection Technology</a>"
html_content12 = f"<h3 style='text-align: center; font-family: 仿宋; color: #117832'>更多最新Yolo-World内容，请点击下方官方文档查看</h3>" \
                f"<a href='https://docs.ultralytics.com/models/yolo-world/' style='display: block; text-align: center'>YOLO-World: Real-Time Open-Vocabulary Object Detection</a>"
html_content13 = f"<img src='{image_url12}' alt='' width='900' style='margin-left: 250px; margin-top: 20px; margin-bottom: 20px'>"
html_content14 = f"<h3 style='text-align: center; font-family: 仿宋; color: #ce1cd4'>更多最新RT-DETR内容，请点击下方官方文档查看</h3>" \
                f"<a href='https://docs.ultralytics.com/models/rtdetr/' style='display: block; text-align: center'>Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector</a>"
html_content15 = f"<img src='{image_url13}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content16 = f"<img src='{image_url14}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content17 = f"<div style='text-align: center; font-family: 仿宋; font-size: large; font-weight: bold; color: black; margin-top: 20px; margin-bottom: 20px'>论文参考：" \
                f"<a href='https://arxiv.org/pdf/2010.03045.pdf'>Rotate to Attend: Convolutional Triplet Attention Module</a></div>" \
                f"<img src='{image_url15}' alt='' width='700' style='margin-left: 350px; margin-bottom: 20px'>"
html_content18 = f"<img src='{image_url16}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content19 = f"<img src='{image_url17}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"
html_content20 = f"<img src='{image_url18}' alt='' width='1100' style='margin-left: 150px; margin-top: 20px; margin-bottom: 20px'>"

def show_train_data_v5():
    return gr.HTML(value=html_content3), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v8():
    return gr.HTML(value=html_content4), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v81():
    return gr.HTML(value=html_content6), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v82():
    return gr.HTML(value=html_content7), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v83():
    return gr.HTML(value=html_content9), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v84():
    return gr.HTML(value=html_content16), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v85():
    return gr.HTML(value=html_content18), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v86():
    return gr.HTML(value=html_content19), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v87():
    return gr.HTML(value=html_content20), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v9():
    return gr.HTML(value=html_content13), gr.update(interactive=False), gr.update(visible=True)

def show_train_data_v91():
    return gr.HTML(value=html_content15), gr.update(interactive=False), gr.update(visible=True)

def close_train_data():
    return gr.HTML(value=""), gr.update(interactive=True), gr.update(visible=False)

def all_choices(all):
    if all:
        return gr.update(value=coco_list)
    else:
        return gr.update(value="")

def all_choices_voc(all):
    if all:
        return gr.update(value=voc_list)
    else:
        return gr.update(value="")

def show_list():
    return gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(interactive=False), gr.update(visible=True)

def close_list():
    return gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), gr.update(interactive=True), gr.update(visible=False)

with gr.Blocks(css=css_style, title="基于Gradio的目标检测可视化平台") as demo:
    html1 = gr.HTML(value=html_content)
    with gr.Row():
        with gr.Accordion(label="YOLO-World: Type Best —— Enables the detection of any object"):
            with gr.Row():
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/05/a4b32dd99c88b379.jpg", scale=2, container=False)
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/05/5b300bcf0568e07c.jpg", scale=1, container=False)
            html2 = gr.HTML(value=html_content12)
            with gr.Tab(label="yolo-worldm.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image9,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directoryw1,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_w1,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="yolo-worldl.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image10,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directoryw2,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_w2,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
    with gr.Row():
        with gr.Accordion(label="RT-DETR: A Vision Transformer-Based Real-Time Object Detector"):
            with gr.Row():
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/09/1ba15bdcd1cc5430.jpg", scale=2, container=False)
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/09/d14b2a8bf94e81e7.jpg", scale=1, container=False)
            html3 = gr.HTML(value=html_content14)
            with gr.Tab(label="rtdetr-l.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image12,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directoryrt1,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_rt1,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="rtdetr-x.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image13,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directoryrt2,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_rt2,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
    with gr.Row():
        with gr.Accordion(label="YOLOv9: Accuracy Best —— Marks a significant advancement in real-time object detection"):
            with gr.Row():
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/05/a05497673b4cb215.png", scale=1, container=False)
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/04/05/5f99b536b3903fc6.png", scale=1, container=False)
            html4 = gr.HTML(value=html_content11)
            with gr.Tab(label="yolov9c.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image7,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory91,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v91,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="yolov9e.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image8,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory92,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v92,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="9cpt-coco128-100epochs"):
                button1 = gr.Button(value="点击查看训练数据", variant="primary")
                out = gr.HTML()
                button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                button1.click(fn=show_train_data_v9, inputs=None, outputs=[out, button1, button1_c])
                button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image11,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory93,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v93,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="9c-coco128-500epochs"):
                button1 = gr.Button(value="点击查看训练数据", variant="primary")
                out = gr.HTML()
                button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                button1.click(fn=show_train_data_v91, inputs=None, outputs=[out, button1, button1_c])
                button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image14,
                        live=True,
                        examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory94,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v94,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
    with gr.Row():
        with gr.Accordion(label="YOLOv8: With attention mechanism as a comparison"):
            with gr.Row():
                gr.Image(value="https://picx.zhimg.com/70/v2-3060d1724433ab64383b789102aea79f_1440w.image?source=172ae18b&biz_tag=Post", scale=6, container=False)
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/03/29/fb8b0e02e4048266.jpg", scale=4, container=False)
            html5 = gr.HTML(value=html_content5)
            with gr.Tab(label="yolov8s.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image1,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory81,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v81,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="yolov8m.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        fn=predict_image2,
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]]
                    )
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory82,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v82,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="8spt-coco128-100epochs"):
                button1 = gr.Button(value="点击查看训练数据", variant="primary")
                out = gr.HTML()
                button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                button1.click(fn=show_train_data_v8, inputs=None, outputs=[out, button1, button1_c])
                button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image3,
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory83,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v83,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="8s-coco128-500epochs"):
                button1 = gr.Button(value="点击查看训练数据", variant="primary")
                out = gr.HTML()
                button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                button1.click(fn=show_train_data_v81, inputs=None, outputs=[out, button1, button1_c])
                button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image4,
                        live=True,
                        examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory84,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v84,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="coco128+注意力机制"):
                with gr.Tab(label="8s-cbam-coco128-500epochs"):
                    html6 = gr.HTML(value=html_content8)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v82, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image5,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory85,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v85,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
                with gr.Tab(label="8s-triplet-coco128-500epochs"):
                    html7 = gr.HTML(value=html_content17)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v84, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image15,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory87,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v87,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
                with gr.Tab(label="8s-ema-coco128-500epochs"):
                    html8 = gr.HTML(value=html_content10)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v83, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image6,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory86,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v86,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
            with gr.Tab(label="VOC+注意力机制"):
                with gr.Tab(label="8spt-cbam-VOC-30epochs"):
                    html9 = gr.HTML(value=html_content8)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v85, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image16,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory88,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v88,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
                with gr.Tab(label="8spt-triplet-VOC-30epochs"):
                    html10 = gr.HTML(value=html_content17)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v86, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image17,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory89,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v89,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
                with gr.Tab(label="8spt-ema-VOC-30epochs"):
                    html11 = gr.HTML(value=html_content10)
                    button1 = gr.Button(value="点击查看训练数据", variant="primary")
                    out = gr.HTML()
                    button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                    button1.click(fn=show_train_data_v87, inputs=None, outputs=[out, button1, button1_c])
                    button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                    with gr.Tab(label="图片"):
                        gr.Interface(
                            inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                            outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                            fn=predict_image18,
                            live=True,
                            examples=[["./ultralytics_new/data/images/plane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                      ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                    with gr.Tab(label="文件夹"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_directory810,
                            inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                    gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                    gr.Slider(minimum=0, maximum=1, value=base_conf),
                                    gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs=gr.Gallery(height=600),
                            # live=True
                        )
                    with gr.Tab(label="视频"):
                        with gr.Blocks():
                            button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                            choices = gr.CheckboxGroup(voc_list, value=['person'], type="index", label="请选择检测类别",
                                                       visible=False)
                            all = gr.Checkbox(label="all", visible=False)
                            all.change(fn=all_choices_voc, inputs=all, outputs=choices)
                            button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                            button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                            button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        gr.Interface(
                            fn=predict_video_v810,
                            inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                            outputs="video",
                            examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                      ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                        )
    with gr.Row():
        with gr.Accordion(label="YOLOv5: A major breakthrough in the YOLO model"):
            with gr.Row():
                gr.Image(value="https://img2.baidu.com/it/u=1572920788,4071073152&fm=253&fmt=auto&app=138&f=PNG?w=1187&h=500", width=700, container=False)
                gr.Image(value="https://s3.bmp.ovh/imgs/2024/03/26/6c7a3ca9fd470293.png", width=700, container=False)
            with gr.Tab(label="yolov5s.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image51,
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory51,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v51,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="yolov5m.pt"):
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image53,
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory53,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v53,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
            with gr.Tab(label="5s-coco128-100epochs"):
                button1 = gr.Button(value="点击查看训练数据", variant="primary")
                out = gr.HTML()
                button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                button1.click(fn=show_train_data_v5, inputs=None, outputs=[out, button1, button1_c])
                button1_c.click(fn=close_train_data, inputs=None, outputs=[out, button1, button1_c])
                with gr.Tab(label="图片"):
                    gr.Interface(
                        inputs=[gr.Image(type="pil"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou)],
                        outputs=["image", gr.Textbox(container=False, elem_classes="c1")],
                        fn=predict_image52,
                        live=True,
                        examples=[["./ultralytics_new/data/images/bus.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou],
                                  ["./ultralytics_new/data/images/zidane.jpg", False, False, 0, 1, 0, 1, 1, 1, base_conf, base_iou]])
                with gr.Tab(label="文件夹"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_directory52,
                        inputs=[gr.File(file_count="directory"), "checkbox", "checkbox",
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=1, value=0), gr.Slider(minimum=0, maximum=1, value=1),
                                gr.Slider(minimum=0, maximum=2, value=1), gr.Slider(minimum=0, maximum=2, value=1),
                                gr.Slider(minimum=0, maximum=1, value=base_conf),
                                gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs=gr.Gallery(height=600),
                        # live=True
                    )
                with gr.Tab(label="视频"):
                    with gr.Blocks():
                        button1 = gr.Button(value="点击选择检测目标类别", variant="primary")
                        choices = gr.CheckboxGroup(coco_list, value=['person'], type="index", label="请选择检测类别",
                                                   visible=False)
                        all = gr.Checkbox(label="all", visible=False)
                        all.change(fn=all_choices, inputs=all, outputs=choices)
                        button1_c = gr.Button(value="收起", visible=False, variant="secondary")
                        button1.click(fn=show_list, inputs=None, outputs=[choices, all, button1, button1_c])
                        button1_c.click(fn=close_list, inputs=None, outputs=[choices, all, button1, button1_c])
                    gr.Interface(
                        fn=predict_video_v52,
                        inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=base_conf), gr.Slider(minimum=0, maximum=1, value=base_iou), choices],
                        outputs="video",
                        examples=[["./ultralytics_new/data/video/test_1.mp4", base_conf, base_iou],
                                  ["./ultralytics_new/data/video/test_2_1.mp4", base_conf, base_iou]]
                    )
# demo.launch()
user_info = [
    ("admin", "password"),
    ("guest", "password"),
    ("ljx", "123456"),
    ("jts", "888888"),
]
demo.launch(auth=user_info,
            auth_message='欢迎登录目标检测可视化平台！',
            share=True
            )
# demo.launch(share=True)
