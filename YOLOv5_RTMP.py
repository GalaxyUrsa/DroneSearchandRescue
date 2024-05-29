# coding=utf-8
import os
import sys
sys.path.append(os.getcwd()+r"\ImageEnhancement")
import torch
import torchvision
import torch.nn as nn
import time
import numpy as np
from math import sin, cos, sqrt, ceil, atan2
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
# from ImageEnhancement import ROP as ROP
from ImageEnhancement import Enhance0121 as ImgEnhance
from graphics.draw import draw_circle, draw_rect
import matplotlib.pyplot as plt
from threading import Thread


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Ensemble(torch.nn.ModuleList):
    '''模型集成'''

    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class YOLOV5(object):
    def __init__(self, conf_thres=0.5,
                 iou_thres=0.45,
                 classes=None,
                 imgsz=640,
                 weights=os.getcwd()+r"\weights\yolov5n-ghost-drone-0412.pt"):
                #  weights=r"F:\PythonWorkspace\VesselNameIdentity\yolov5-master\weights\yolov5n6-ghost-drone-0412.pt"):
        # weights="weights/best_line_name_2.22.pt"):
        # 超参数设置
        self.conf_thres = conf_thres  # 置信度阈值
        self.iou_thres = iou_thres  # iou阈值
        self.classes = classes  # 分类个数
        self.imgsz = imgsz  # 归一化大小
        # Load model
        self.device = torch.device('cuda')
        self.model = self.attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = self.check_img_size(imgsz, s=self.stride)  # check img_size
        self.detect_result = []
        self.detect_result_save = []
        # 调用ROP图像增强对象
        # self.rop = ROP.Doing()
        # else:
        #     print(type(frame))
        #     print(frame)
        #     frame = frame.float()  # uint8 to fp16/32
        #     frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        #     frame = rop.demo_enhancement4cvpr2021_revised(frame)

    def clear_detect_result(self):
        self.detect_result.clear()

    def clear_detect_result_save(self):
        self.detect_result_save.clear()

    def attempt_load(self, weights, map_location=None):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(w, map_location=map_location)  # load
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble

    def make_divisible(self, x, divisor):
        # Returns x evenly divisible by divisor
        return ceil(x / divisor) * divisor
        # return math.ceil(x / divisor) * divisor

    def check_img_size(self, img_size, s=32):
        # Verify img_size is a multiple of stride s
        new_size = self.make_divisible(img_size, int(s))  # ceil gs-multiple
        if new_size != img_size:
            print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
        return new_size

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False,
                            labels=()):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def infer(self, image, agnostic_nms=False):
        # read image
        # image=cv2.imread(img_path)

        # Padded resize
        img = self.letterbox(image, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=agnostic_nms)

        # Process detections
        s = ""
        s += '%gx%g ' % img.shape[2:]  # print string
        result = []
        confidence = []
        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                result.append([x1, y1, x2, y2])
                confidence.append(round(float(conf), 3))
                # self.detect_result.append([x1, y1, x2, y2])

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {img_path}{'s' * (n > 1)}, "  # add to string

                # Write results
                # Get names and colors
                names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # 推理结束后把位置记录下来
                    pos = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    draw_circle(image, ((pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2),
                        int(sqrt((pos[0] - pos[2])**2 + (pos[1] - pos[3])**2) / 2))
                    draw_rect(image, pos)
        return image, result, confidence

    def infer_skip(self, image, _pos, agnostic_nms=False):
        # read image
        img = self.letterbox(image, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        for xyxy in _pos:
            # self.plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=3)
            pos = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            draw_circle(image, ((pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2),
                        int(sqrt((pos[0] - pos[2])**2 + (pos[1] - pos[3])**2) / 2))

                        # int(math.sqrt(math.pow(pos[0] - pos[2], 2) + math.pow(pos[1] - pos[3], 2)) / 2))
            draw_rect(image, pos)
        return image

# 计算经纬度坐标
def coordinate(height, FOV, midx, midy, x, y, CL, n_width=1920, lon=29, lat=114):
    """
    Args:
        height:无人机飞行高度
        FOV: 无人机视场角
        midx: 检测框中心坐标x
        midy: 检测框中心坐标y
        x: 检测框的左上检测点坐标x
        y: 检测框的左上检测点坐标y
        CL: 航向角
        n_width: 分辨率
        lon: 当前纬度
        lat: 当前经度
    Returns:
    """
    gamma = degrees(270 - CL)
    # gamma = math.degrees(270 - CL)
    factor_transfer = np.array([[cos(gamma), -sin(gamma)], [sin(gamma), cos(gamma)]])
    # factor_transfer = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
    xy = np.array([[x - midx], [y - midy]])
    # print("factor:{}".format(factor_transfer))
    # print("xy:{}".format(xy))
    _xy = factor_transfer @ xy
    _xy = _xy.tolist()
    # print("_xy:{}".format(_xy))

    l = sqrt((x - midx) * (x - midx) + (y - midy) * (y - midy))
    # l = math.sqrt((x - midx) * (x - midx) + (y - midy) * (y - midy))
    pixel = (2 * height * tan(FOV / height) / n_width)
    # pixel = (2 * height * math.tan(FOV / height) / n_width)
    alpha = 1 / ((40030173 * cos(lon * 3.1415927 / 180)) / 360)
    # alpha = 1 / ((40030173 * math.cos(lon * 3.1415927 / 180)) / 360)
    beta = 1 / (40030173 / 360)
    theta = atan2(_xy[0][0], _xy[1][0])
    # theta = math.atan2(_xy[0][0], _xy[1][0])
    result = [lon + alpha * l * cos(theta) * pixel, lat + beta * l * sin(theta) * pixel]
    # result = [lon + alpha * l * math.cos(theta) * pixel, lat + beta * l * math.sin(theta) * pixel]
    return result


# 处理视频
def process_video(input_file, output_file):
    """
    Args:
        input_file: 要处理的视频的地址
        output_file: 要输出的视频的地址
    """
    # 调用ROP图像增强对象
    # rop = ROP.Doing()
    # 记录处理的帧数(主要是用于跳帧，后面有详细解释)
    infer_num = 0
    # 文字样式，默认微软雅黑，编码格式为gbk
    font = ImageFont.truetype("yahei.ttf", 27, encoding="gbk")  # word
    # YOLOv5对象
    yolo = YOLOV5()
    # 读取视频流
    cap = cv2.VideoCapture(input_file)
    # 获取视频画面的分辨率、帧率等数据
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 开启VideoWriter， 用于保存处理后的视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    # 用于存放检测后的结果
    detect_result = []
    # 用于存放位置(xyxy)、置信度(conf)等结果
    _pos, _conf = [], []
    # 只要视频还没有结束，循环不断
    while cap.isOpened():
        t1 = time.time()
        # 读取帧和这一帧的数据，好像是这样的
        ret, frame = cap.read()
        frame = ImgEnhance.Enhance(frame)
        if not ret:
            break
        if (infer_num == 0):
            # 读取到当前帧后，开始检测(推理inference)
            frame, pos, conf = yolo.infer(frame)
            # 用于存放检测结果的detect_result清零，然后加上检测得到的位置信息
            detect_result.clear()
            detect_result.append(pos)
            # write info on the image
            # 让当前帧转换为一个Image对象，方便后期在上面打印位子
            pilimg = Image.fromarray(frame)  # word
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            # 从detect_result中逐个读取检测框坐标信息，开始逐个处理
            # 好像detect_result怎么搞都是只有一个的(每次都清零)，无所谓，写的时间有点长不记得了
            for xyxy in detect_result:  # result of detection
                for num in range(len(xyxy)):  # circle for result
                    # midx=(x1+x2)/2  midy=(x2+y2)/2
                    """
                    xyxy:
                    x1   y1    x2    y2
                    0    1     2     3
                    """
                    midx, midy = int((xyxy[num - 1][0] + xyxy[num - 1][2]) / 2), int(
                        (xyxy[num - 1][1] + xyxy[num - 1][3]) / 2)
                    """
                    上    下    左     右
                    up  down  left  right
                    """
                    # 检测框的上下左右，这里主要是用于计算检测框外圆
                    up, down, left, right = xyxy[num - 1][1], xyxy[num - 1][3], xyxy[num - 1][0], xyxy[num - 1][2]
                    # width, height分别是画面的宽和高
                    width = frame_width
                    height = frame_height
                    # 检测框外圆半径
                    radius = sqrt((up - down) * (up - down) / 4 + (left - right) * (left - right) / 4)
                    # radius = math.sqrt((up - down) * (up - down) / 4 + (left - right) * (left - right) / 4)
                    # 检测中心点经纬度坐标
                    coord = coordinate(100, 83, height / 2, width / 2, midx, midy, 0, 2720, 30.609011, 114.354996)
                    string = "发现遇险人员\n" + "CONF:" + str(conf[num - 1])[:4]

                    if (((height - down) >= 300) and (left >= 200) and ((width - right) >= 200)):
                        draw.text((midx - radius - 50, midy + radius + 20), string, (0, 0, 0), font=font)
                    else:
                        draw.text((midx + radius + 20, midy - radius), string, (0, 0, 0), font=font)
            # Image对象转换为cv2对象，即矩阵转换成三通道RGB
            frame = cv2.cvtColor(np.array(pilimg)[..., ::-1].copy(), cv2.COLOR_BGR2RGB)
            _pos, _conf = pos, conf
            infer_num += 1
        else:
            # detect_result.clear()
            # 记录这一时刻的位置和置信度，在后面的跳帧检测中使用
            pos, conf = _pos, _conf
            # YOLOv5对象中的跳帧
            frame = yolo.infer_skip(frame, pos)
            # write info on the image
            pilimg = Image.fromarray(frame)  # word
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            for xyxy in detect_result:  # result of detection
                for num in range(len(xyxy)):  # circle for result
                    midx, midy = int((xyxy[num - 1][0] + xyxy[num - 1][2]) / 2), int(
                        (xyxy[num - 1][1] + xyxy[num - 1][3]) / 2)
                    up, down, left, right = xyxy[num - 1][1], xyxy[num - 1][3], xyxy[num - 1][0], xyxy[num - 1][2]
                    width = frame_width
                    height = frame_height
                    radius = sqrt((up - down) * (up - down) / 4 + (left - right) * (left - right) / 4)
                    # radius = math.sqrt((up - down) * (up - down) / 4 + (left - right) * (left - right) / 4)
                    coord = coordinate(50, 83, height / 2, width / 2, midx, midy, 0, 2720, 30.609011, 114.354996)
                    string = "CONF:" + str(conf[num - 1]) + "\n" + \
                             "LON :" + str(round(coord[0], 6)) + "\n" + \
                             "LAT :" + str(round(coord[1], 6))
                    string = "发现遇险人员\n" + "CONF:" + str(conf[num - 1])[:4]
                    # 下面
                    # 左右200pixel，下面400pixel
                    if (((height - down) >= 300) and (left >= 200) and ((width - right) >= 200)):
                        draw.text((midx - radius - 50, midy + radius + 20), string, (0, 0, 0), font=font)
                    else:
                        draw.text((midx + radius + 20, midy - radius), string, (0, 0, 0), font=font)

            frame = cv2.cvtColor(np.array(pilimg)[..., ::-1].copy(), cv2.COLOR_BGR2RGB)
            # frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_BGR2RGB)
            _pos, _conf = pos, conf
            infer_num += 1
            if (infer_num == 2):
                infer_num = 0

        # 输出检测后的视频
        # out.write(frame)
        # 实时画面显示窗口名
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("result", 640, 640)
        cv2.resizeWindow("Frame", 1920, 1080)
        # 实时画面窗口分辨率
        # cv2.resizeWindow("Frame", 300, 300)
        cv2.imshow('Frame', frame)
        t2 = time.time()
        print("deltaT:", t2-t1)
        # 强制退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # print(coordinate(100, 83, 300, 300, 0, 0, 180))

    input_file = "video/drone_1.MP4"
    output_file = "drone_1.mp4"
    process_video(input_file, output_file)