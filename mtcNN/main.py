import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.ops.boxes import batched_nms
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
        if pretrained:
            state_dict_path = './.pt/pnet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        classification = self.conv4_1(x)
        probability = self.softmax4_1(classification)
        regression = self.conv4_2(x)
        return regression,probability

class RNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        if pretrained:
            state_dict_path = './.pt/rnet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        # print(x.shape) >>> torch.Size([617, 64, 3, 3])
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        classification = self.dense5_1(x)
        probability = self.softmax5_1(classification)
        regression = self.dense5_2(x)
        return regression, probability

class ONet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)
        if pretrained:
            state_dict_path = './.pt/onet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)  # 3*48*48->32*46*46
        x = self.prelu1(x)
        x = self.pool1(x)  # 32*46*46->32*23*23
        x = self.conv2(x)  # 32*23*23->64*21*21
        x = self.prelu2(x)
        x = self.pool2(x)  # 64*21*21->64*10*10
        x = self.conv3(x)  # 64*10*10->64*8*8
        x = self.prelu3(x)
        x = self.pool3(x)  # 64*8*8->64*4*4
        x = self.conv4(x)  # 64*4*4->128*3*3
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))  # 128*9=1152->256
        x = self.prelu5(x)
        classification = self.dense6_1(x)  # 256->2
        probability = self.softmax6_1(classification)
        regression = self.dense6_2(x)  # 256->4
        landmark = self.dense6_3(x)  # 256->10
        return regression, landmark, probability

def rectangleToSquare(boxes):
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]

    l = torch.max(h,w)
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - l * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - l * 0.5
    boxes[:, 2] = boxes[:, 0] + l
    boxes[:, 3] = boxes[:, 1] + l

    return boxes

def cropImg(boxes,image_inds,imgs,size):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 0] = 0
    y[y < 0] = 0
    ex[ex > w-1] = w - 1
    ey[ey > h-1] = h - 1
    data = []
    for k in range(len(y)):
        if (ex[k]-x[k]>1 and ey[k]-y[k]>0):
            img = imgs[image_inds[k],:,y[k]:ey[k],x[k]:ex[k]].unsqueeze(0)
            img = torch.nn.functional.interpolate(img, size=size)
            data.append(img)
    data = torch.cat(data,dim=0)
    return data
# 定义超参数???
BATCH = 16
pyramid_factor = 0.5 # 图像金字塔缩放比例
video_scale = 0.25
THRESHOLD=[0.6,0.7,0.7]
DEVICE = 'cuda'
# 实例化模型
pnet = PNet()
rnet = RNet()
onet = ONet()
# 打开视频
test_path = './demo.mp4'
#print("视频路径就是：\n",test_path)
video = cv2.VideoCapture(test_path)  # 读取视频（没有声音只有图像）
v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 读取视频的长度，视频有v_len帧的图片组成
sample = np.arange(0,v_len)
ImageToTensor = transforms.ToTensor()
faces = []
frames = []
landmark_x = []
landmark_y = []
for j in range(v_len):
    success,frame = video.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize([int(frame.size[0]*video_scale),int(frame.size[1]*video_scale)])
    # print(ImageToTensor(frame))
    frames.append(torch.Tensor(np.array(frame)).unsqueeze(0))
    # frames.append(frame)
    print(len(frames))
    if len(frames) % BATCH == 0 or j == v_len-1:
        imgs = torch.cat(frames,dim=0) # 组合成一个batch
        imgs = imgs.permute((0,3,1,2)) # imgs.shape=torch.Size([16, 3, 270, 480])
        imgs_1 = imgs
        batch_size = len(imgs)
        h, w = imgs.shape[2:4]

        # 创建尺度金字塔
        scale_i = pyramid_factor
        minl = min(h, w)
        scales = []  # 存放要检测不同大小的人脸的像素相对于整张图的比例
        while True:
            if(minl*pyramid_factor>12):
                scales.append(scale_i)
                minl = minl*pyramid_factor
                scale_i = scale_i * pyramid_factor
            else:
                break

        boxes = []
        image_inds = []
        for scale in scales:

            data = torch.nn.functional.interpolate(imgs,size=(int(h*scale),int(w*scale)))
            data = (data-127.5)/128
            regression, prob = pnet(data) # regression.shape=[16,4,63,115]，prob.shape=[16,2,63.115]
            reg = regression.permute(0,2,3,1) # reg.shape=[4,16,63,116]
            prob = prob[:,1] # prob[:,0] :不是人脸的概率，prob{:,1}:是人脸的概率
            mask = prob >= THRESHOLD[0] # 是人脸的概率大于阈值
            mask_ind = mask.nonzero() # mask_ind.shape=[853.3]
            # 计算候选框
            points = mask_ind[:,1:3].flip(1)
            points_right_down = ((points*2)/scale).floor() # 候选框从特征图映射回原图上
            points_left_up = ((points*2+12)/scale).floor()
            boxes.append(torch.cat([points_right_down,points_left_up,prob[mask].unsqueeze(1),reg[mask,:]],dim=1))
            image_inds.append(mask_ind[:, 0])
        boxes = torch.cat(boxes,dim=0) # 把不同尺度的候选框放在一起,总共1189个候选框
        image_inds = torch.cat(image_inds,dim=0)
        # 非极大抑制
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick] # 只剩下589个候选框
        # 边框回归
        regw = boxes[:,2] - boxes[:,0]
        regh = boxes[:,3] - boxes[:,1]
        point1 = (boxes[:, 0] + boxes[:, 5] * regw).unsqueeze(1)
        point2 = (boxes[:, 1] + boxes[:, 6] * regh).unsqueeze(1)
        point3 = (boxes[:, 2] + boxes[:, 7] * regw).unsqueeze(1)
        point4 = (boxes[:, 3] + boxes[:, 8] * regh).unsqueeze(1)
        boxes = torch.cat([point1,point2,point3,point4,boxes[:,4].unsqueeze(1)],dim=1)
        boxes = rectangleToSquare(boxes)
        data = cropImg(boxes,image_inds,imgs,(24,24)) # data.shape=[589,3,24,24]
        data = (data-127.5)/128
        # 剪裁出来的图片放到RNet中
        reg,prob = rnet(data)
        prob = prob[:,1]
        mask = prob >= THRESHOLD[0]
        boxes = torch.cat([boxes[mask,:4],prob[mask].unsqueeze(1)],dim=1) # boxes.shape=[208,5]
        image_inds,reg = image_inds[mask],reg[mask]
        # 再一次NMS
        pick = batched_nms(boxes[:,:4],boxes[:,4],image_inds,0.7)
        boxes,image_inds,reg=boxes[pick],image_inds[pick],reg[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        point1 = (boxes[:, 0] + reg[:,0] * regw).unsqueeze(1)
        point2 = (boxes[:, 1] + reg[:,1] * regh).unsqueeze(1)
        point3 = (boxes[:, 2] + reg[:,2] * regw).unsqueeze(1)
        point4 = (boxes[:, 3] + reg[:,3] * regh).unsqueeze(1)
        boxes = torch.cat([point1, point2, point3, point4, boxes[:, 4].unsqueeze(1)], dim=1)
        boxes = rectangleToSquare(boxes)
        # 根据候选框，把候选框图片从原图中剪裁出来,作为onet的输入
        data = cropImg(boxes, image_inds, imgs, (48, 48))  # data.shape=[103,3,48,48]
        data = (data - 127.5) / 128
        # 把剪裁出来的图片放到onet中
        reg,landmark,prob = onet(data)
        prob = prob[:,1]
        mask = prob >= THRESHOLD[2]
        reg,landmark,boxes,image_inds = reg[mask],landmark[mask],boxes[mask],image_inds[mask]
        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        point1 = (boxes[:, 0] + reg[:, 0] * regw).unsqueeze(1)
        point2 = (boxes[:, 1] + reg[:, 1] * regh).unsqueeze(1)
        point3 = (boxes[:, 2] + reg[:, 2] * regw).unsqueeze(1)
        point4 = (boxes[:, 3] + reg[:, 3] * regh).unsqueeze(1)
        boxes = torch.cat([point1, point2, point3, point4, boxes[:, 4].unsqueeze(1)], dim=1)
        # landmark
        points_x = regw.repeat(5,1).permute(1,0) * landmark[:, :5] + boxes[:, 0].repeat(5, 1).permute(1,0)
        points_y = regh.repeat(5,1).permute(1,0) * landmark[:, 5:10] + boxes[:, 1].repeat(5, 1).permute(1,0)
        landmarks = torch.cat([points_x,points_y],dim=1)
        # NMS
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes,image_inds = boxes[pick],image_inds[pick]

        # 得到的候选框boxes，得到的候选框属于的照片image_inds,得到的关键点landmarks
        boxes = boxes.cpu().detach().numpy()
        landmarks = landmarks.cpu().detach().numpy()
        for b_i in range(batch_size):
            b_i_inds = np.where(image_inds == b_i)
            # batch_boxes.append(boxes[b_i_inds].copy())
            faces.append(boxes[b_i_inds].copy())
            landmark_y.append(points_y[b_i_inds])
            landmark_x.append(points_x[b_i_inds])
        frames = []
video.release()
video = cv2.VideoCapture(test_path)  # 读取视频（没有声音只有图像）
v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 读取视频的长度，视频有v_len帧的图片组成
fps_video = video.get(cv2.CAP_PROP_FPS) #获取视频帧率
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #获取视频宽度
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #获取视频高度
videoWriter = cv2.VideoWriter("result.mp4", fourcc, fps_video, (frame_width, frame_height))
print(landmark_y)
for i in range(v_len):
    success,frame = video.read()
    for box in faces[i]:
        print(box)
        cv2.rectangle(frame,(int(box[0]/video_scale),int(box[1]/video_scale)),
                      (int(box[2]/video_scale),int(box[3]/video_scale)),(55,255,155),5)
    videoWriter.write(frame)
videoWriter.release()
