import os
import concurrent.futures
import multiprocessing as mp
import numpy as np
from PIL import Image,ImageOps
Image.MAX_IMAGE_PIXELS = None

train_orig=os.path.join('.', 'train_wk.txt')
valid_orig=os.path.join('.', 'valid_wk.txt')

with open(os.path.join('.', 'train.txt'), 'w') as trainfile:
    with open(train_orig, 'r') as ft:

        img_pathes=[line.replace('\n','') for line in ft.readlines()]

        for i, img_path in enumerate(img_pathes):
            print(f'{(100*i)/len(img_pathes)}%')
            img = Image.open(img_path).convert('RGB')
            H=img.height
            W=img.width
            label_path=img_path.replace('images','labels').replace('jpg','txt')
            with open(label_path,'r') as fl:
                labels=[list(map(float,line.replace('\n','').split(' ')) )for line in fl.readlines()]
                if labels==[]:
                    continue
            labelLine=os.path.split(img_path)[1]
            for label in labels:
                labelLine+=' '
                label_id=int(label[0])
                center_x, center_y=label[1]*W, label[2]*H
                bw, bh=label[3]*W, label[4]*H
                xmin=int(center_x-(bw/2))
                xmax=int(center_x+(bw/2))
                ymin=int(center_y-(bh/2))
                ymax=int(center_y+(bh/2))
                lbstr=f'{xmin},{ymin},{xmax},{ymax},{label_id}'#data\custom\Image\USBA_20200304163548538539.jpg 4030,184,4526,1231,0 1375,465,2108,1403,1
                labelLine +=lbstr
            trainfile.write(labelLine+'\n')
            #if i>3:
             #  break

with open(os.path.join('.', 'valid.txt'), 'w') as validfile:
    with open(valid_orig, 'r') as fv:

        img_pathes=[line.replace('\n','') for line in fv.readlines()]

        for i, img_path in enumerate(img_pathes):
            print(f'{(100*i)/len(img_pathes)}%')
            img = Image.open(img_path).convert('RGB')
            H=img.height
            W=img.width
            label_path=img_path.replace('images','labels').replace('jpg','txt')
            with open(label_path,'r') as fl:
                labels=[list(map(float,line.replace('\n','').split(' ')) )for line in fl.readlines()]
                if labels==[]:
                    continue
            labelLine=os.path.split(img_path)[1]
            for label in labels:
                labelLine+=' '
                label_id=int(label[0])
                center_x, center_y=label[1]*W, label[2]*H
                bw, bh=label[3]*W, label[4]*H
                xmin=int(center_x-(bw/2))
                xmax=int(center_x+(bw/2))
                ymin=int(center_y-(bh/2))
                ymax=int(center_y+(bh/2))
                lbstr=f'{xmin},{ymin},{xmax},{ymax},{label_id}'#data\custom\Image\USBA_20200304163548538539.jpg 4030,184,4526,1231,0 1375,465,2108,1403,1
                labelLine +=lbstr
            validfile.write(labelLine+'\n')
            #if i>3:
             #  break

    
