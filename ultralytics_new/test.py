from PIL import Image
from ultralytics import YOLO
from ultralytics import RTDETR
conf_thres = 0.25
iou_thres = 0.45

model = YOLO('yolov9c.pt')
# model = RTDETR('rtdetr-l.pt')
# results = model('data/images/bus.jpg')
# results = model.predict(source='data/images/zidane.jpg', conf=conf_thres, iou=iou_thres, save=True, save_conf=True, save_txt=True, name='output')
results = model.predict(source='data/images/tennis.jpg', conf=conf_thres, iou=iou_thres, classes=['person'])
im_array = results[0].plot()
pil_img = Image.fromarray(im_array[..., ::-1])
pil_img.save('data/result.jpg')
ans = "识别到图片中物体及置信度如下：\n"
count = 1
# while i < len(results[0].boxes.cls):
for i in range(len(results[0].boxes.cls)):
    ans = ans + results[0].names[int(results[0].boxes.cls[i])] + str(count) +":conf=" + str(round(float(results[0].boxes.conf[i]),3)) + "    "
    # print("%s%d %.3f" % (results[0].names[int(results[0].boxes.cls[i])], count, float(results[0].boxes.conf[i])), end='   ')
    count += 1
    # print(results[0].names[int(results[0].boxes.cls[i])], round(float(results[0].boxes.conf[i]),3))
    if i + 1 < len(results[0].boxes.cls) and results[0].boxes.cls[i + 1] != results[0].boxes.cls[i]:
        count = 1
        ans = ans + "\n"
print(ans)
        # print("")
# model = YOLO('yolov8s.pt')
# results = model.predict(source="data/video/test_1.mp4", save=True)
# import cv2
# from ultralytics import YOLO
# # 加载模型
# model = YOLO('yolov8s.pt')
# # 打开视频文件
# video_path = "data/video/test_1.mp4"
# cap = cv2.VideoCapture(video_path)
# # 获取视频帧的维度
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# #创建VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('data/video/output1.mp4', fourcc, 20.0, (frame_width, frame_height))
# #循环视频帧
# while cap.isOpened():
#     # 读取某一帧
#     success, frame = cap.read()
#     if success:
#         # 使用yolov8进行预测
#         results = model(frame)
#         #可视化结果
#         annotated_frame = results[0].plot()
#         #将带注释的帧写入视频文件
#         out.write(annotated_frame)
#     else:
#         # 最后结尾中断视频帧循环
#         break
# #释放读取和写入对象
# cap.release()
# out.release()
