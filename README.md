Drone and Rescue v1.0
Release in May 22nd
==============================================
/************系统概要************/

/************配置、部署及使用步骤************/
1)安装Python编译器(基于Anaconda/Miniconda)
conda create -n Drone python=3.9
2)进入Drone环境
conda activate Drone
3)进入Drone根目录
cd ~/Drone
4)安装所依赖的Python包
pip install -r requirements.txt
5)检测后端代码是否可正常运行
cd ~/Background
python main.py
6)检测前后端代码是否可正常连接并运行
cd ~/FlaskDemo
python app.py


/************目录结构描述************/
Drone                     //无人机辅助搜救系统
├── Background            //系统后端代码
│   ├── graphics          //图形处理
│   ├── lmageEnhancement  //图像增强
│   ├── logs              //日志记录   
│   ├── models            //目标检测模型
│   ├── Nginx             //基于Nginx的反向代理服务器
│   ├── weights           //图像增强网络、目标检测网络权重
│   ├── main.py           //后端处理主函数
│   └── Detect_RTMP.py    //检测RTMP视频流数据
├── FlaskDemo             //基于Flask框架的前端测试用例
│   ├── view.html         //生成特定内容的网页模板
│   └── app.py            //启动测试用例
├── requirements.txt      //项目所依赖的Python包及版本
├── LICENSE               //项目需遵循的开源协议
└── ReadMe.txt            //项目概述、安装与配置说明、使用指南

/************LICENSE************/
DroneSearchandRescue is licenced under GPLv3.

The source of target Detection part is from https://github.com/ultralytics/yolov5/.

Copyright © 2024 by Jianheng Huang, Yunqing Wang.
==============================================
