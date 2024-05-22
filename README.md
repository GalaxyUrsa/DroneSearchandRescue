Drone and Rescue v1.0<br>
Release in May 23rd<br>

---
I 配置、部署及使用步骤<br>
1)安装Python编译器(基于Anaconda/Miniconda)<br>
conda create -n Drone python=3.9<br>
2)进入Drone环境<br>
conda activate Drone<br>
3)进入Drone根目录<br>
cd ~/Drone<br>
4)安装所依赖的Python包<br>
pip install -r requirements.txt<br>
5)检测后端代码是否可正常运行<br>
cd ~/Background<br>
python main.py<br>
6)检测前后端代码是否可正常连接并运行<br>
cd ~/FlaskDemo<br>
python app.py<br>

---
II 目录结构描述<br>
Drone                     //无人机辅助搜救系统<br>
├── Background            //系统后端代码<br>
│   ├── graphics          //图形处理<br>
│   ├── lmageEnhancement  //图像增强<br>
│   ├── logs              //日志记录<br>
│   ├── models            //目标检测模型<br>
│   ├── Nginx             //基于Nginx的反向代理服务器<br>
│   ├── weights           //图像增强网络、目标检测网络权重<br>
│   ├── main.py           //后端处理主函数<br>
│   └── Detect_RTMP.py    //检测RTMP视频流数据<br>
├── FlaskDemo             //基于Flask框架的前端测试用例<br>
│   ├── view.html         //生成特定内容的网页模板<br>
│   └── app.py            //启动测试用例<br>
├── requirements.txt      //项目所依赖的Python包及版本<br>
├── LICENSE               //项目需遵循的开源协议<br>
└── ReadMe.txt            //项目概述、安装与配置说明、使用指南<br>

---
LICENSE<br>
DroneSearchandRescue is licenced under GPLv3.<br>
The source of target Detection part is from https://github.com/ultralytics/yolov5/.<br>
Copyright © 2024 by Jianheng Huang, Yunqing Wang.<br>
