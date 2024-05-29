import sys
import os
import cv2
from cv2 import resize, imencode
import threading
import time
import win32gui, win32con
import YOLOv5_RTMP
from ImageEnhancement import Enhance0121 as ImgEnhance
import time
from cv2 import cvtColor
import numpy as np
from math import sqrt
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import time
import subprocess
from flask import Flask, Response, render_template
from flask_cors import CORS
import queue
from pynput import keyboard
from plugin.litsen import DestoryThread

sys.path.append(r"F:\PythonWorkspace\yolov5-Drone\background")
sys.path.append(os.getcwd()+r"\ImageEnhancement")
sys.path.append(os.getcwd()+r"\RTMP")

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})
yolo = YOLOv5_RTMP.YOLOV5()

class Producer(threading.Thread):
    """docstring for Producer"""
    def __init__(self, rtmp_str):
        super(Producer, self).__init__()
        self.rtmp_str = rtmp_str
        self.Nginx = self.startNginx()
        # YOLOv5对象
        self.yolo = yolo
        self.font = ImageFont.truetype(os.getcwd()+r"\yahei.ttf", 27, encoding="gbk")  # word
        # 通过cv2中的类获取视频流操作对象cap
        self.cap = cv2.VideoCapture(self.rtmp_str)
        # 调用cv2方法获取cap的视频帧（帧：每秒多少张图片
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 获取cap视频流的每帧大小
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)
        print(self.size)
        # 视频队列
        self.video_queue = queue.Queue()
        # 卡帧
        self.key_frame = None
        self._running = True
        self._litsener_thread = None
        self.valueset = [50, 50, 50, 0]

    def run(self):
        """
        This function is the main function of the class method Producer
        """
        self._litsener_thread = threading.Thread(target=self._start_litsener)
        self._litsener_thread.start()
        t_start = time.time()
        frame_num = 0
        print('in producer')
        ret, frame = self.cap.read()
        # frame = ImgEnhance.Enhance(frame)
        y = 0
        t0 = time.time()
        while self._running:
            if ret:
                frame_num += 1
                ts1 = time.time()
                frame = resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                ts2 = time.time()
                frame = self.infer(frame)
                ts3 = time.time()
                _, buffer = imencode('.jpg', frame)
                TCP_Frame = buffer.tobytes()
                self.key_frame = TCP_Frame
                t1 = time.time()
                t0 = time.time()
                ret, frame = self.cap.read()
                ret, frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
                continue
        print("thread is ended!")
        self.__del__()

    def infer(self, frame):
        """
        This function is for infering frame

        :param frame: The RTMP frame in the type of ndarray need to be infered by yolo
        :return: The infered RTMP frame in the type of ndarray
        """
        frame, pos, conf = self.yolo.infer(frame)
        return frame

    # Start Nginx Processor
    def startNginx(self):
        """
        This function is for starting Nginx server by subprocess

        :return: Method of Nginx server
        """
        exe_path = os.getcwd() + r"\RTMP\nginx-rtmp-win64-master\nginx.exe"
        print(exe_path)
        Nginx = subprocess.Popen(exe_path)
        print("run Nginx...")
        return Nginx

    # Return Video Stream in the type of Bytes
    def retrunVideoStream(self):
        """
        This function is for returning infered RTMP frame in the type of Bytes from video_queue to panel

        :return: Return RTMP frame in the type of Bytes
        """
        while True:
            if self.key_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.key_frame + b'\r\n')    
            else:
                continue

    def _start_litsener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key.char=='q':
                self._running = False
                self._litsener_thread.join()
                # self.__del__()
                print("is destoryed")
            if key.char=='u' and self.valueset[0]<200:
                self.valueset[0]+=1
                print(self.valueset)
            if key.char=='h' and self.valueset[1]>1:
                self.valueset[1]-=1
                print(self.valueset)
            if key.char=='j' and self.valueset[1]<200:
                self.valueset[1]+=1
                print(self.valueset)
            if key.char=='n' and self.valueset[2]>15:
                self.valueset[2]-=1
                print(self.valueset)
            if key.char=='m' and self.valueset[2]<100:
                self.valueset[2]+=1
            if key.char=='o':
                # print(key.char, valueset[3])
                if self.valueset[3] == 1:
                    self.valueset[3] = 0
                else:
                    self.valueset[3] = 1
            print(self.valueset)
        except AttributeError:
            print(f"type error")

    def __del__(self):
        """
        This function is for destorying class method Producer
        """
        # self.Nginx.kill()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Nginx is killed!\nProgram is done!") 

@app.route('/')
def index():
    return render_template('view.html')

@app.route('/video_feed')
def video_feed():
    # rtmp_str = 'rtmp://10.91.61.49:1935/live/hls'
    # rtmp_str = 'rtmp://192.168.31.247/live/hls'
    rtmp_str = 'rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid'
    processStream = Producer(rtmp_str)
    processStream.start()
    return Response(processStream.retrunVideoStream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def runServer():
    app.run(debug=False, use_reloader=False)

if __name__ == '__main__':
    print('run program')
    # app.run(debug=True)
    server_thread = threading.Thread(target=runServer)
    server_thread.start()
    server_thread.join()
    print('end!')