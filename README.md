Drone and Rescue v1.0<br>
Release in May 23rd<br>

---
系统概要<br>
配置、部署及使用步骤<br>
安装Python编译器(基于Anaconda/Miniconda)<br>
conda create -n Drone python=3.9<br>
进入Drone环境<br>
conda activate Drone<br>
进入Drone根目录<br>
cd ~/Drone<br>
安装所依赖的Python包<br>
pip install -r requirements.txt<br>
检测后端代码是否可正常运行<br>
cd ~/Background<br>
python main.py<br>
检测前后端代码是否可正常连接并运行<br>
cd ~/FlaskDemo<br>
python app.py<br>

---
目录结构描述
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

---
LICENSE

DroneSearchandRescue is licenced under GPLv3.

The source of target Detection part is from https://github.com/ultralytics/yolov5/.

Copyright © 2024 by Jianheng Huang, Yunqing Wang.
