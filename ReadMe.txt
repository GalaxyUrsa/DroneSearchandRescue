Drone and Rescue v1.0
Release in May 23rd
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

遵循 GNU General Public License v3.0 协议，标明目标检测部分来源：https://github.com/ultralytics/yolov5/



       

├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools

├── Panel                 //系统前端代码
│   ├──CCS                //
│   ├──datas              //
│   ├── images            //
│   ├── JS                //
│   ├── 无人机1.0_slices  //
│   ├── 飞行轨迹          //
│   ├── demo.html         //
│   ├── flaskProject.rar  //
│   ├── flyHistory.html   //
│   ├── fullScreen.html   //
│   ├── wflyHistory.html  //
│   └── wmainFrame.html   //


###########V1.0.0 版本内容更新
1. 新功能     aaaaaaaaa
2. 新功能     bbbbbbbbb
3. 新功能     ccccccccc
4. 新功能     ddddddddd



Drone and Rescue v1.0
Release in May 23rd
系统概要
配置、部署及使用步骤
安装Python编译器(基于Anaconda/Miniconda)
bash
conda create -n Drone python=3.9
进入Drone环境
bash
conda activate Drone
进入Drone根目录
bash
cd ~/Drone
安装所依赖的Python包
bash
pip install -r requirements.txt
检测后端代码是否可正常运行
bash
cd ~/Background  
python main.py
检测前后端代码是否可正常连接并运行
bash
cd ~/FlaskDemo  
python app.py
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
LICENSE
DroneSearchandRescue is licenced under GPLv3.

The source of target Detection part is from https://github.com/ultralytics/yolov5/.

Copyright © 2024 by Jianheng Huang, Yunqing Wang.

以上是您的内容用Markdown重新排版的结果。希望这可以帮助到您!


目录结构描述

         Drone                               // 无人机辅助搜救系统  
         ├──── Background                    // 系统后端
         │     ├──── graphics                // 图形处理  
         │     ├──── ImageEnhancement        // 图像增强模型 
         │     ├──── logs                    // 日志记录     
         │     ├──── models                  // 目标检测模型  
         │     ├──── Nginx                   // 基于Nginx的反向代理服务器  
         │     ├──── weights                 // 图像增强网络、目标检测网络权重  
         │     ├──── main.py                 // 后端处理主函数  
         │     └──── Detect_RTMP.py          // 检测RTMP视频流数据  
         ├──── Panel                         // 系统前端  
         │     ├──── CCS                     // 系统界面预设
         │     ├──── datas                   // 存放系统运行过程产生的数据
         │     ├──── images                  // 组成系统界面的图片
         │     ├──── JS                      // 系统前端响应
         │     ├──── main.html               // 系统主界面
         │     ├──── flyHistory.html         // 历史数据查询界面
         │     ├──── fullScreen.html         // AR飞行界面
         │     └──── wmainFrame.html         // 实时视频显示界面
         └──── requirements.txt              // 项目所需的依赖包及版本  


