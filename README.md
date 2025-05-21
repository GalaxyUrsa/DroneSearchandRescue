# Current Version:
### Drone and Rescue v1.2
- Release on Aug 21st
# History:
### Drone Search and Rescue v1.1
- Release on May 29th
### Drone Search and Rescue v1.0
- Release on May 23rd
---
# I. Configuration, deployment, and usage procedures

### (1) Install Python compiler (based on Anaconda/Miniconda)
```bash
conda create -n Drone python=3.9
```
### (2) Enter the Drone environment
```bash
conda activate Drone
```
### (3) Enter the root of the project
```
cd ~/Drone
```
### (4) Install the required package
```bash
pip install -r requirements.txt
```
### (5) Verify the back-end code
```
cd ~/Background
```
```bash
python main.py
```
### (6) Verify that back-end code is connected and running properly
```bash
python app.py
```
---
# II. Directory Structure
```
Drone                     //Drone Search and Rescue
├── Background            //Back-end Code
│   ├── conf              //Configuration of Nginx
│   ├── graphics          //Process graphics
│   ├── ImageEnhancement  //Image enhancement(Updating now. Please delete the relevant code in main.py and YOLOv5_RTMP.py)
│   ├── logs              //Record log
│   ├── models            //Target detection models
│   ├── RTMP              //RTMP server refer to Nginx
│   ├── utils             //Utility of YOLOv5
│   ├── weights           //Weights for neural network(Please prepare by yourself and modify its path in YOLOv5_RTMP.py)
│   ├── main.py           //main function
│   ├── temp              //Create the temp folder and create a subfolder hls. Otherwise, it cannot run
│   └── YOLOv5_RTMP.py    //Detect RTMP stream
├── FlaskDemo             //Test case based on Flask
│   ├── view.html         //Display template on HTML
│   └── app.py            //Test case for application
├── requirements.txt      //Required package
├── LICENSE               //License
├── font.ttf              //The font type you want to display on the panel
└── ReadMe.txt            //Operating guide
```
---
# I. 配置、部署及使用步骤

### (1) 安装Python编译器(基于Anaconda/Miniconda)
```bash
conda create -n Drone python=3.9
```
### (2) 进入Drone环境
```bash
conda activate Drone
```
### (3) 进入Drone根目录
```
cd ~/Drone
```
### (4) 安装所依赖的Python包
```bash
pip install -r requirements.txt
```
### (5) 检测后端代码是否可正常运行
```
cd ~/Background
```
```bash
python main.py
```
### (6) 检测前后端代码是否可正常连接并运行
```bash
python app.py
```
---
# II. 目录结构描述
```
Drone                     //无人机辅助搜救系统
├── Background            //系统后端代码
│   ├── conf              //Nginx配置
│   ├── graphics          //图形处理
│   ├── ImageEnhancement  //图像增强(更新中，如运行不成功请自行在main.py与YOLOv5_RTMP.py文件中注释用于import模块的语句)
│   ├── logs              //日志记录
│   ├── models            //目标检测模型
│   ├── RTMP              //基于Nginx的反向代理服务器(请自行解压)
│   ├── utils             //YOLOv5目标检测网络所需插件(来源于yolov5源码)
│   ├── weights           //图像增强网络、目标检测网络权重(权重请自行准备，可以使用公开的yolov5n/s/m/l.pt等，在YOLOv5_RTMP.py文件中修改)
│   ├── main.py           //后端处理主函数
│   ├── temp              //自行创建temp文件夹，并建子文件夹hls，否则无法运行
│   └── Detect_RTMP.py    //检测RTMP视频流数据
├── FlaskDemo             //基于Flask框架的前端测试用例
│   ├── view.html         //生成特定内容的网页模板
│   └── app.py            //启动测试用例
├── requirements.txt      //项目所依赖的Python包及版本
├── LICENSE               //项目需遵循的开源协议
├── font.ttf              //项目需使用一种字体，需自行准备(font替换成自己的文件名)
└── ReadMe.txt            //项目概述、安装与配置说明、使用指南
```
---
# LICENSE
- DroneSearchAndRescue is licensed under the MIT License
- The source of target Detection part is from https://github.com/ultralytics/yolov5/.
- Copyright © 2024 by Jianheng Huang, Yunqing Wang, Ruoling Lai.
---
# Contact Us
- Email: [jianhenghuang26@gmail.com](jianhenghuang26@gmail.com)
---
# Acknowledgement
- This project is the source code for us to participate in the competition. We only released the back-end part and part of the code based on the Flask framework. Due to the large amount of work, we did not release the front-end part.
- I would like to thank team members Yunqing Wang and Ruoling Lai for their contributions and support.
- We completed this project at Wuhan University of Technology during 2023-2024.