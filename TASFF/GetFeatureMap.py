# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:46:03 2021

@author: 612_public
"""

#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized
 


# 训练过的模型路径
#resume_path = r"D:\TJU\GBDB\set113\cross_validation\test1\epoch_0257_checkpoint.pth.tar"
# 输入图像路径
single_img_path = r'E:\vessel_detection\ASFF-4-SFPN\yolov5-master\data\try2\11.tif'
# 绘制的热力图存储路径
save_path = r'E:\vessel_detection\ASFF-4-SFPN\yolov5-master\runs\feature\temp.jpg'
 
# 网络层的层名列表, 需要根据实际使用网络进行修改
#layers_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
# 指定层名
#out_layer_name = "layer4"
 
features_grad = 0


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    #img = img.unsqueeze(0)
 
    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)
 
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    

if __name__ == '__main__':
    modelc = load_classifier(name='resnet101', n=2)  # initialize
   # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    draw_CAM(modelc, single_img_path, save_path, transform=None, visual_heatmap=True)