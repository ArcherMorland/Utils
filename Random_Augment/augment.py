import os, random, math, uuid, copy, time

from datetime import datetime
import multiprocessing
import concurrent.futures
import xml.etree.ElementTree as ET

import glob
import numpy as np
from torchvision import transforms

from PIL import Image,ImageOps
Image.MAX_IMAGE_PIXELS = None

#import matplotlib.pyplot as plt

import concurrent.futures
import multiprocessing as mp
import threading

import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def cfg_parse():
    cfg_content=dict()
    with open(os.path.join('.','augment.cfg'),'r') as cfg:
        lines=[l.replace(' ','').replace('\n','') for l in cfg.readlines()]

    for line in lines:
        if line == '' or line.startswith('#'):
            continue
        k,v=line.split('=')
        cfg_content.update({k:v})
        
    return cfg_content


cfg=cfg_parse()
with open(os.path.normpath(cfg["classes"]),'r') as cf:
    #classes=[c.replace('\n','') for c in cf.readlines()]
    classes=dict()
    classes_lines=[cs.replace(' ','').split(',') for cs in [c.replace('\n', '') for c in cf.readlines()]]
    for v, clist in enumerate(classes_lines):
        for k in clist:
            classes.update({k.lower():v})

Limiting_Computation_Resources=True
CompRsc= 2 if Limiting_Computation_Resources else os.cpu_count()#the number of applied cores


def show(imgT):
    plt.imshow(imgT.permute(1,2,0))
    plt.show()


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


class ImageData:
    def __init__(self, imgPath):
        global classes,cfg
        
        self.imgPath=imgPath
        img=Image.open(imgPath).convert('RGB')
        container, filename = os.path.split(imgPath)
        self.annotationPath=imgPath.replace('Image','XML').replace('.jpg','.xml')
        
        self.Info=dict()
        self.labels=dict()
        
        ImgId = np.base_repr(uuid.uuid4().int, base=35).rjust(25,"Z")
        
        with open( self.annotationPath) as in_file:
            self.Info.update({ "image_stamp" : f"{filename}-{ImgId}"})

            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            
            self.w = img.size[0] #int(size.find('width').text) 
            self.Info.update({'width': self.w})
            
            self.h = img.size[1] #int(size.find('height').text)
            self.Info.update({'height': self.h})
            img=None
            
            for obj in root.iter('object'):
                cls = obj.find('name').text.lower()
                difficult = int(obj.find('difficult').text)
                if cls not in classes or difficult == 1:
                    continue
                
                
                TargetObj=dict()

                #cls_id = classes.index(cls)
                cls_id = classes[cls]
                xmlbox = obj.find('bndbox')
               
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((self.w, self.h), b)
                
                Label_ID=np.base_repr(uuid.uuid4().int, base=35).rjust(25,"Z")
                
                TargetObj.update({"label_id" : Label_ID})
                TargetObj.update({"class" : cls})

                TargetStamp=f"{cls}-{Label_ID}"
                
                TargetObj.update({"class_id" : cls_id})
                TargetObj.update({"difficult":difficult})
                TargetObj.update({ "xmin_xml":float(xmlbox.find('xmin').text) })
                TargetObj.update({ "xmax_xml":float(xmlbox.find('xmax').text) })
                TargetObj.update({ "ymin_xml":float(xmlbox.find('ymin').text) })
                TargetObj.update({ "ymax_xml":float(xmlbox.find('ymax').text) })
                
                TargetObj.update({ "x_centor_txt":bb[0] })
                TargetObj.update({ "y_center_txt":bb[1] })
                TargetObj.update({ "box_width_txt":bb[2] })
                TargetObj.update({ "box_height_txt":bb[3] })

                TargetObj.update({ "image_path":self.imgPath})

                self.labels.update({ TargetStamp : TargetObj})
                
            self.Info.update({ "labels" : self.labels})
            
    

    
    def Stamp_SysLoading(self):
        Stamp = datetime.now().strftime('%Y/%m/%d-%H:%M:%S.%f')
        return Stamp


    def getTensor(self):
        return transforms.ToTensor()(Image.open(self.imgPath).convert('RGB'))





def Merge_Pictures(backgroundPath, combination, guests_info, annotation_path):
    global cfg

    #annopath, annotxt=os.path.split(cfg["new_train_annotation_path"])
    #print(annotation_path)
    #print(p)
    annopath, annotxt = os.path.split(annotation_path)
    print(os.path.split(annotation_path))
    #if not os.path.exists(annopath):
     #   os.makedirs(annopath)
    os.makedirs(annopath, exist_ok=True)   
    #if not os.path.exists(cfg["new_data_container"]):
     #   os.makedirs(cfg["new_data_container"])
    os.makedirs(cfg["new_data_container"], exist_ok=True)



    
    new_annotation_content=list()
    '''
    new_annotation_container,fname = os.path.split(cfg["new_annotation_path"])
    fname_split = fname.split('.')

    new_fname=fname_split[0]+f"_{np.base_repr(uuid.uuid4().int, base=35).rjust(25,'Z')}."+fname_split[1]

    new_annotation_path=os.path.join(new_annotation_container, new_fname)
    
    with open(new_annotation_path,'w') as f:
        for i,c in enumerate(combination):
            background_path = random.sample(backgroundPath,1)
            regions_position = dict()
            for ind in c:
                regions_position.update({guests_info[ind]["label_id"]:{"path":guests_info[ind]["image_path"],
                                                                   "box":(guests_info[ind]["xmin_xml"],
                                                                          guests_info[ind]["ymin_xml"],
                                                                          guests_info[ind]["xmax_xml"],
                                                                          guests_info[ind]["ymax_xml"]),
                                                                   "classname":guests_info[ind]["class"],
                                                                   "classid":guests_info[ind]["class_id"]
                                                                   }})
            merge_record=Merge_Picture(regions_position, background_path)
            
            
            print(merge_record)
            f.write(merge_record+'\n')'''
    for i,c in enumerate(combination):
        background_path = random.sample(backgroundPath,1)
        regions_position = dict()
        for ind in c:
            regions_position.update({guests_info[ind]["label_id"]:{"path":guests_info[ind]["image_path"],
                                                                   "box":(guests_info[ind]["xmin_xml"],
                                                                          guests_info[ind]["ymin_xml"],
                                                                          guests_info[ind]["xmax_xml"],
                                                                          guests_info[ind]["ymax_xml"]),
                                                                   "classname":guests_info[ind]["class"],
                                                                   "classid":guests_info[ind]["class_id"]
                                                                   }})
        merge_record=Merge_Picture(regions_position, background_path)
            
        
        print(merge_record)
        
        new_annotation_content.append(merge_record+'\n')
        #time.sleep(3)
    return new_annotation_content

def Merge_Picture(regions, backgroud):

    results=list()
    region_images=[Image.open(dict_value["path"]).convert('RGB').crop(dict_value["box"]) for dict_value in regions.values()]
    #random.shuffle(region_images)
    new_region_pos=copy.deepcopy(regions)

    background_image=Image.open(backgroud[0]).convert('RGB')
    half_width = background_image.width//2
    half_height = background_image.height//2
    
    for key in new_region_pos.keys():
        region=Image.open(new_region_pos[key]["path"]).convert('RGB').crop(new_region_pos[key]["box"])
        
        if region.width > half_width:
            ratio=region.height/region.width
            tw=round(half_width*0.9)
            th=round(tw*ratio)
            region=region.resize((tw,th), Image.BILINEAR)
            
        if region.height>half_height:
            
            ratio=region.width/region.height
            th=round(half_height*0.9)
            tw=round(th*ratio)
            
            region=region.resize((tw,th), Image.BILINEAR)
            
        new_region_pos[key].update({"region_image":  region })
        
    Quad_order=[1,2,3,4]
    padding = (10, 10, 10, 10)


    
    
    
    if True:#region_images>=3:
        random.shuffle(Quad_order)
        for q, key in enumerate(new_region_pos.keys()):
            
            r = ImageOps.expand(new_region_pos[key]["region_image"], padding, fill='white')
            
            
            x_offset = random.random()*(half_width-10-10)
            y_offset = random.random()*(half_height-10-10)

            box=list( map( int, (10+x_offset, 10+y_offset, 10 +r.width +x_offset, 10+r.height + y_offset ) ))
            if box[2] >=half_width:
                box[0]=half_width-10-r.width
                box[2]=half_width-10
                
            if box[3] >=half_height:
                box[1]=half_height-10-r.height
                box[3]=half_height-10
                
            
            #================================
            if Quad_order[q]==1:
                box[0]+=half_width
                box[2]+=half_width
                   
            elif Quad_order[q]==3:               
                box[1]+=half_height
                box[3]+=half_height
                               
            elif Quad_order[q]==4:
                box[0]+=half_width
                box[2]+=half_width
                box[1]+=half_height
                box[3]+=half_height
                
            
            background_image.paste(r, box)
            
            new_region_pos[key]['box']=list(map(lambda x:str(x-10) if box.index(x)>1 else str(x + 10) , box))
            items=copy.deepcopy(new_region_pos[key]['box'])
            items.append(str(new_region_pos[key]["classid"]))
            
            results.append(items)
            #r.show()
        
        save_name=f"{cfg['type']}_{np.base_repr(uuid.uuid3(uuid.NAMESPACE_DNS, uuid.uuid1().hex).int, base=35).rjust(25,'Z')}.jpg"#
        #pic_ucode=uuid.uuid1()
        #save_name=os.path.join(cfg["new_data_container"],f"{cfg['type']}_{np.base_repr(uuid.uuid3(pic_ucode, uuid.uuid1().hex).int, base=35).rjust(25,'Z')}.jpg") 
        background_image.save(os.path.join(cfg["new_data_container"], save_name), quality=100)
        
    results=list(map(lambda x:','.join(x), results))
    #print(save_name)
    #print(p)
    return os.path.relpath(save_name,  os.curdir)+" "+" ".join(results)


def BackgroundBased_Merge(hosts_container=None, guests_container=None, background_container=None, annotation_path=None, num_save=5000):

    from collections import ChainMap
    global cfg, CompRsc, Limiting_Computation_Resources
    
    if guests_container==None:
        hosts_path=glob.glob(os.path.join(hosts_container,"*.jpg"))
        guests_path=copy.deepcopy(hosts_path)

    
    backgrounds_path=glob.glob(os.path.join(background_container,"*.jpg"))
    
    guests_info=dict()
    guests_grouping=dict()
    #test_guests_grouping=dict()
    for c in classes:
        guests_grouping.update({c:list()})
        #test_guests_grouping.update({c:list()})
    #====================================================
    '''
    t1=time.time()
    for path in guests_path:
        #t1=time.time()
        obj=ImageData(path)
        #t2=time.time()-t1
        #print(t2)
        
        guests_info.update(obj.labels)#could be separated
        
        for key in obj.labels.keys():#could be separated
            guests_grouping[obj.labels[key]["class"]].append(key)
    t2=time.time()-t1
    
    print("normal:   ",t2)
    #print("guests_info:　　　",guests_info)
    #print("guests_grouping:   ",guests_grouping)
    print('\n\n\n\n')'''
    #====================================================
    #t1=time.time()
    
    #mw=os.cpu_count()*2-1
    mw=CompRsc*2-1
    with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as executor:
        results = executor.map(ImageData, guests_path)
        result_list=[obj.labels for obj in results]
        guests_info=dict(ChainMap(*result_list))
        
    for key in guests_info.keys():
        guests_grouping[guests_info[key]["class"]].append(key)
        
    #t2=time.time()-t1
    #print("MT:   ",t2)
    #print("guests_info II:   ",guests_info)
    #print("test_guests_grouping:   ",test_guests_grouping)
    #print(p)
    #====================================================
            
    cb=list()
    for key in guests_grouping.keys():#為每一種label產生至少5000 or num_save張圖片
        for i in range(num_save):
            cb.append(sum( [list(random.sample(guests_info.keys(),random.randint(0,3))), list(random.sample(guests_grouping[key],1))], []))
            
    #print("checkpoints:",backgrounds_path)
    #Normal:
    #t1=time.time()
    #Merge_Pictures(backgrounds_path, cb, guests_info)
    #print("Normal takes:   ", time.time()-t1)
    #multi-processing:
    t1=time.time()
    #chunks=[cb[i:i+(len(cb)//os.cpu_count())] for i in range(0, len(cb), len(cb)//os.cpu_count() )]
    chunks=[cb[i:i+(len(cb)//CompRsc)] for i in range(0, len(cb), len(cb)//CompRsc )]
    backgrounds_path_rep=[ backgrounds_path for i in range(len(chunks))]
    guests_info_rep=[guests_info for i in range(len(chunks))]
    annotation_pathes=[annotation_path for i in range(len(chunks))]
    
    #with mp.Pool(os.cpu_count()) as pool:
    with mp.Pool(CompRsc) as pool:
        results=pool.starmap_async(Merge_Pictures, zip(backgrounds_path_rep, chunks, guests_info_rep, annotation_pathes))
        collection = results.get()

        
    
    annotation=sum(collection,[])
    #with open(cfg["new_annotation_path"],'w') as f:
    with open(annotation_path, 'w') as f:
        for line in annotation:
            f.write(line)
        
    #print("annotation:     ",annotation)
    print("MP takes:   ", time.time()-t1)
    print("cb len:     ", len(cb))
    #Merge_Pictures(backgrounds_path, cb, guests_info)
    #multi-threading:
        
    return "fininshed!!"

'''
Classlist_path=os.path.join('.','Origin','SOD','Classes_SOD.txt')

    imgPath=os.path.join('.','Origin', 'SOD', 'Image', 'Data_Basic_SOD_Type_B_0001.jpg')

    Img=ImageData(imgPath)

    labels=[key for key in Img.Info['labels'].keys() if key.split('-')[0] in classes]

    labelObj=Img.Info["labels"][labels[0]]
    
    box1=(labelObj["xmin_xml"], labelObj["ymin_xml"], labelObj["xmax_xml"], labelObj["ymax_xml"])
    
    image1=Image.open(imgPath).convert('RGB')
    region1 = image1.crop(box1)
    print(region1.size)
    #add border to image
    padding = (10, 10, 10, 10)
    region1 = ImageOps.expand(region1, padding, fill='white')
    print(region1.size)
    #region1.show()


imgPath_chair=os.path.join('.','Origin', 'SOD', 'Image', 'Data_Basic_SOD_Type_B_0181.jpg')
    image2=Image.open(imgPath_chair).convert('RGB')
    tw=1992-1435
    th=1847-976
    region1=region1.resize((tw,th), Image.BILINEAR)
    
    image2.paste(region1, (1435, 976, 1992, 1847))
    image2.save("merge_test.jpg",quality=100)
    image2.show()
    
#==========
    
import glob
import os


files=glob.glob(os.path.join('.','XML_orig','*.xml'))

with open('Classes_Chip.name','r') as cf:
    #classes=[c.replace('\n','') for c in cf.readlines()]
    classes_c=dict()
    classes_c_lines=[cs.replace(' ','').split(',') for cs in [c.replace('\n', '') for c in cf.readlines()]]
    for v, clist in enumerate(classes_c_lines):
        for k in clist:
            classes_c.update({k.lower():v})


with open('classes.name','r') as cf:
    #classes=[c.replace('\n','') for c in cf.readlines()]
    classes=dict()
    classes_lines=[cs.replace(' ','').split(',') for cs in [c.replace('\n', '') for c in cf.readlines()]]
    for k, clist in enumerate(classes_lines):
        for v in clist:
            classes.update({k:v})



#classes_c
#{'basic_chip_a_top': 0,
#'basic_chip_b_top': 0,
 #'basic_chip_a_side_long': 1,
 #'basic_chip_b_side_long': 1,
 #'basic_chip_a_side_short': 2,
 #'basic_chip_b_side_short': 2, 
 #'basic_chip_a_pattern': 3,
 #'basic_chip_b_pattern': 3,
 #'basic_chip_b_bottom': 4}


#classes
# {0: 'Basic_Chip_Top',
#  1: 'Basic_Chip_Side_Long',
#  2: 'Basic_Chip_Side_Short',
 # 3: 'Basic_Chip_Pattern',
  #4: 'Basic_Chip_Bottom'}


os.makedirs(os.path.join('.', 'XML'), exist_ok=True)

for fp in files:
    with open(fp,'r') as f:
        lines=f.readlines()

    
    with open(fp.replace('XML_orig','XML'),'w')as fw:
        for ln in lines:       
            for c in classes_c.keys():
                if c in ln.lower():
                    ln=ln.lower().replace(c,classes[classes_c[c]])
            fw.write(ln)

#=============================================================================
type=PLCC
classes = ./original/PLCC/classes.name
hosts_container = ./original/PLCC/Image/
background_container=./original/PLCC/Background/
new_data_container = ./NewData/PLCC/Image/
new_train_annotation_path = ./NewData/PLCC/PLCC_train.txt
new_valid_annotation_path = ./NewData/PLCC/PLCC_valid.txt
'''
if __name__ == '__main__':

    total_gen_per_target=5000
    split_size=0.2
    validation_size=round(total_gen_per_target*split_size)
    training_size=total_gen_per_target-validation_size

    train_msg=BackgroundBased_Merge(hosts_container=cfg["hosts_container"],
                                    background_container=cfg["background_container"],
                                    annotation_path=cfg["new_train_annotation_path"],
                                    num_save=training_size)
    print('train',train_msg)
    valid_msg = BackgroundBased_Merge(hosts_container=cfg["hosts_container"],
                                      background_container=cfg["background_container"],
                                      annotation_path=cfg["new_valid_annotation_path"],
                                      num_save=validation_size)
    print('valid',valid_msg)


    
    






#https://www.itread01.com/content/1549998912.html
#https://www.itread01.com/content/1565193664.html
