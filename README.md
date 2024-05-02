# 2024届毕业设计基于Gradio的目标检测可视化平台
简介:本项目基于Gradio开发了一个目标检测可视化平台，支持多种模型、数据集和评估指标，并提供图片、文件夹、视频等预处理检测，提供丰富的可视化结果展示方式。该平台可用于轻松评估和探索目标检测模型的性能，并支持本地部署和远程访问。
## 1. 安装
安装gradio库:
<pre>pip install gradio==4.22.0</pre>
## 2. 准备数据
### 下载预训练权重模型：
从 https://docs.ultralytics.com/models/ 网站中下载预训练权重模型yolov5su.pt、yolov5mu.pt、yolov8s.pt、yolov8m.pt、yolov9c.pt、yolov9e.pt、yolov8m-worldv2.pt、yolov8l-worldv2.pt、rtdetr-l.pt、rtdetr-x.pt。
### 解压模型：
将下载的模型解压到 ultralytics_new 目录下。
## 3. 运行脚本
运行脚本：
<pre>python gradio_try.py</pre>
## 4. 使用平台
### 访问平台：
+ 在浏览器中打开 http://localhost:7860 即可访问平台。
+ 在程序运行界面找到 public URL ，在浏览器中输入远程访问平台。
### 登录平台：
在登录界面输入已有的用户名和密码即可登录平台。
|用户名|密码|
|:----:|:----:|
|admin|password|
|guest|password|
|ljx|123456|
|jts|888888|
### 进行检测：
登录平台后即可选择模型进行图片、视频等的预处理和检测。
