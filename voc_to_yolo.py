#from yolo_dataset import getTraVal_list

import os, random, shutil,copy
from os.path import *
import glob
from lxml.etree import Element, SubElement, tostring, ElementTree
import xml.etree.ElementTree as ET

from PIL import Image
import torch
import torch.utils.data as data

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets

import matplotlib.pyplot as plt


def show(imgT):
    plt.imshow(imgT.permute(1,2,0))#PyTorch Tensors ("Image tensors") are channel first, so to use them with matplotlib user need to reshape it
                                   #https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch/53633017
                                   #像素顺序是RGB(critical reference!!!!): https://www.jianshu.com/p/c0ba27e392ff
                                   #to gray_scale:https://stackoverflow.com/questions/52439364/how-to-convert-rgb-images-to-grayscale-in-pytorch-dataloader
    plt.show()



#TrainList, ValList=getTraVal_list()

with open(join('.','classes.name')) as cf:
    #classes=[c.replace('\n', '') for c in cf.readlines()]
    classes=dict()
    classes_lines=[cs.replace(' ','').split(',') for cs in [c.replace('\n', '') for c in cf.readlines()]]
    for v, clist in enumerate(classes_lines):
        for k in clist:
            classes.update({k.lower():v})



def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)
 
 
# 處理單一XML檔案
def convert_annotation(image_add):
    # image_add进来的是带地址的.jpg
    img=Image.open(image_add).convert('RGB')
    image_name = os.path.split(image_add)[1]# 取檔案名稱
    print(image_name)
    image_name = image_name.replace('.jpg', '')  # 刪除副檔名
    
    in_file = open( join( *(normpath('./XML').split(os.sep)) ,image_name +'.xml'))  # 取得圖檔對應的標註檔
    out_file = open(join( normpath('./labels'), image_name+'.txt'), 'w')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
 
    size = root.find('size')
 
    w = img.size[0]# int(size.find('width').text)
    h = img.size[1]# int(size.find('height').text)
 
    # 迭代所有出現在XML檔的object
    for obj in root.iter('object'):
        # iter()方法可以递归遍历元素/树的所有子元素
        difficult = obj.find('difficult').text
        cls = obj.find('name').text.lower()
        # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
        if cls not in classes or int(difficult) == 1:
            print(cls)
            print(p)
            continue
        #cls_id = classes.index(cls)#这里取索引，避免类别名是中文，之后运行yolo时要在cfg将索引与具体类别配对
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
 
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
if not os.path.exists(join('.','labels')):#不存在文件夹
    os.makedirs(join('.','labels') )
 
image_adds = glob.glob(join('.','Image','*.jpg'))#open(join('.','custom','train.txt'))
for image_add in image_adds:
    image_add = image_add.strip()
    convert_annotation(image_add)
 
print("Finished")

























