"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
import cv2
import torch
import torch.utils.data as dataf

import matplotlib.pyplot as plt
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

City_Name= (
           'Bejing', 'Berling', 'Busan', 'LA', 'Lundon', 'Paris', 'Seoul', 'Shanghai', 'Sydney', 'Tokyo','New York')
##至少在dataloader中存放200张图片
class CityToGTADataloader():

    def __init__(self,cfg,classname):
        super(CityToGTADataloader,self).__init__()

        self.rootpath=cfg['DataSet']['filepath']## train dataset path and test dataset path
        self.img_size=cfg['DataSet']['img_size']

    def Create_Data(self):

        for city in City_Name:
            city_path=os.path.join(self.rootpath,str(city))
            n = len(os.listdir(city_path))/2
            city_matrix=np.zeros((n,4))
            save_path = self.rootpath + '' + str(city) + '/'
            isExists = os.path.exists(save_path)
            if not isExists:
                os.mkdir(save_path)
            for i in range(n):
                index="%03d" % i
                xml_file=os.path.join(city_path,''+str(index)+'.xml')
                img_file=os.path.join(city_path,''+str(index)+'.jpg')
                if os.path.exists(xml_file) and os.path.exists(img_file):
                    root = ET.parse(xml_file).getroot()
                    for entire in root.iter('object'):
                        if entire.find('name').text == 'car':
                            box = entire.find('bndbox')
                            box = [float(box.find('xmin').text), float(box.find('ymin').text), float(box.find('xmax').text),
                                   float(box.find('ymax').text)]
                            img_path = self.rootpath + '00' + str(i) + '.jpg'
                            img = cv2.imread(img_path, 1)
                            origin_size = [img.shape[0], img.shape[1]]
                            out_size = [self.img_size, self.img_size]
                            resize_box = self.box_resize(box, origin_size, out_size)
                            img = cv2.resize(img, (256, 256))
                            ymin = int(resize_box[1] - 1)
                            ymax = int(resize_box[3] - 1)
                            xmin = int(resize_box[0] - 1)
                            xmax = int(resize_box[2] - 1)
                            instance = img[ymin:ymax, xmin:xmax, :]
                            city_matrix[i]=resize_box
                            cv2.imwrite(save_path+''+str(i)+'_label.jpg',instance)
                i += 1

            np.save(save_path+'/'+str(city)+'.npy', city_matrix)
    def create_label(self):
        path=self.rootpath+'test'
        n = len(os.listdir(path)) / 2
        matrix=np.zeros((n,4))
        for i in range(0,n):
            index="%05d"%i
            xml_file=os.path.join(path,''+str(index)+'.xml')
            img_file=os.path.join(path,''+str(index)+'.jpg')
            if os.path.exists(xml_file) and os.path.exists(img_file):
                root = ET.parse(xml_file).getroot()
                for entire in root.iter('object'):
                    if entire.find('name').text == 'car':
                        box = entire.find('bndbox')
                        box = [float(box.find('xmin').text), float(box.find('ymin').text), float(box.find('xmax').text),
                               float(box.find('ymax').text)]
                        img_path = self.rootpath + '00' + str(i) + '.jpg'
                        img = cv2.imread(img_path, 1)
                        origin_size = [img.shape[0], img.shape[1]]
                        out_size = [self.img_size, self.img_size]
                        resize_box = self.box_resize(box, origin_size, out_size)
                        img = cv2.resize(img, (256, 256))
                        ymin = int(resize_box[1] - 1)
                        ymax = int(resize_box[3] - 1)
                        xmin = int(resize_box[0] - 1)
                        xmax = int(resize_box[2] - 1)
                        instance = img[ymin:ymax, xmin:xmax, :]
                        matrix[i]=resize_box
                        cv2.imwrite(path+''+str(i)+'_label.jpg',instance)
            i += 1

        np.save(path+'/label.npy', matrix)

    def __load_class(self,class_name):## the name of images which contains class
        """Load individual image indices from splits."""
        #print(class_name)
        ids_A = []
        ids_B=[]
        root = os.path.join(self.rootpath, 'VOCdevkit/VOC2012/ImageSets')
        root = os.path.join(root,'Main', class_name[0] + '_train.txt')
        #print(root)
        with open(root, 'r') as f:
            for element in f.readlines():
                element=element.split()
                #print(element[1])
                if element[1]=='1':## 所有正例图片id
                    #print(element)
                    ids_A+=[element[0]]
        root = os.path.join(self.rootpath, 'VOCdevkit/VOC2012/ImageSets')
        root = os.path.join(root, 'Main', class_name[1] + '_train.txt')
        with open(root, 'r') as f:
            for element in f.readlines():
                element = element.split()
                if element[1] == '1':
                    ids_B += [element[0]]
        return ids_A,ids_B




    def box_resize(self,box,original_size,out_size):
        if not len(original_size) == 2:
            raise ValueError("original size requires length 2 tuple, given {}".format(len(original_size)))
        if not len(out_size) == 2:
            raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))
        #print(original_size)
        #print(out_size)
        bbox = box.copy()
        y_scale = out_size[0] / original_size[0]
        x_scale = out_size[1] / original_size[1]

        bbox[1] =max(min(np.round(bbox[1]*y_scale),out_size[1]),0)
        bbox[3] =max(min(np.round(y_scale * bbox[3]),out_size[1]),0)
        bbox[0] = max(min(np.round(bbox[0]*x_scale),out_size[0]),0)
        bbox[2] = max(min(np.round(x_scale * bbox[2]),out_size[0]),0)

        return bbox
    ##根据图片中已有的object制作相对应的identity matrix并对图像进行分割
    def load__label(self,name,ids,mean=[0.4914, 0.4822, 0.4465]):
        #ids=self.img_array##  保存所属图片的名称
        matrix_size=np.zeros((len(ids),4))
        i=0
        for imgname in ids:
            path=os.path.join(self.rootpath,'VOCdevkit/VOC2012/Annotations/',''+str(imgname)+'.xml')

            root = ET.parse(path).getroot()
            size=root.find('size')
            height=int(size.find('height').text)
            width=int(size.find('width').text)
            #print(height)
            entire=root.find('object')
            #@print(entire.dtext())
            for entire in root.iter('object'):
                if entire.find('name').text==name:
                    box = entire.find('bndbox')
                    box=[float(box.find('xmin').text),float(box.find('ymin').text),float(box.find('xmax').text),float(box.find('ymax').text)]
                    img_path=self.rootpath+'VOCdevkit/VOC2012/JPEGImages/'+str(imgname)+'.jpg'
                    img=cv2.imread(os.path.join(self.rootpath,'VOCdevkit/VOC2012/JPEGImages/',''+str(imgname)+'.jpg'),1)

                    origin_size=[img.shape[0],img.shape[1]]
                    out_size=[256,256]
                #resize_box=box
                    resize_box = self.box_resize(box,origin_size,out_size)

                #print(resize_box)
                #img_2=self.resize_image(img,512)
                    img=cv2.resize(img,(256,256))
                #print(img.shape)
                    Identity_matrix=np.zeros(img.shape)

                    ymin=int(resize_box[1]-1)
                    ymax=int(resize_box[3]-1)
                    xmin=int(resize_box[0]-1)
                    xmax=int(resize_box[2]-1)

                #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),[0,0,120],3)
                    Identity_matrix[ymin:ymax,xmin:xmax,:]=1
                #Identity_matrix=[resize_box[1]-1:resize_box[3]-1,resize_box[0]-1:resize_box[2]-1,:]=1
                    instance=img[ymin:ymax,xmin:xmax,:]
                    save_path=self.rootpath+'file/'+str(name)+'/'+str(imgname)+'/'
                    isExists=os.path.exists(save_path)
                    #print(save_path)
                    if not isExists:
                        os.makedirs(save_path)
                    cv2.imwrite(save_path+''+str(imgname)+'com.jpg', img)
                    np.save(save_path+''+str(imgname)+'.npy',Identity_matrix)
                    cv2.imwrite(save_path + '' + str(imgname) + 'com_label.jpg', instance)
                    if img.shape[2]==3:
                        img[ymin:ymax,xmin:xmax,0]=mean[0]
                        img[ymin:ymax, xmin:xmax, 1] = mean[1]
                        img[ymin:ymax, xmin:xmax, 2] = mean[2]
                    else:
                        img[ymin:ymax, xmin:xmax, 0] = mean[0]
                    if img.shape[2] == 3:
                        img[ymin:ymax, xmin:xmax, 0] = img[:,:,0].mean()
                        img[ymin:ymax, xmin:xmax, 1] = img[:,:,1].mean()
                        img[ymin:ymax, xmin:xmax, 2] = img[:,:,2].mean()
                    else:
                        img[ymin:ymax, xmin:xmax, 0] = mean[0]
                    cv2.imwrite(save_path + '' + str(imgname) + 'com_pretreated_mean.jpg', img)
                #print(matrix_size.shape)
                    matrix_size[i] = resize_box
            i+=1

        np.save(self.rootpath+'file/'+name+'/final.npy',matrix_size)

    def train_loader(self,name):
        original_path="/root/UNIT/Data/"
        isExists=os.path.exists(original_path+'train_A')
        if not isExists:
            os.makedirs(original_path+'train_A')
        isExists=os.path.exists(original_path+'train_B')
        if not isExists:
            os.makedirs(original_path+'train_B')
        isExists=os.path.exists(original_path+'test_A')
        if not isExists:
            os.makedirs(original_path+'test_A')
        isExists=os.path.exists(original_path+'test_B')
        if not isExists:
            os.makedirs(original_path+'test_B')
        isExists=os.path.exists(original_path+'train_ins_a')
        if not isExists:
            os.makedirs(original_path+'train_ins_a')
        isExists=os.path.exists(original_path+'train_ins_b')
        if not isExists:
            os.makedirs(original_path+'train_ins_b')
        isExists=os.path.exists(original_path+'test_ins_a')
        if not isExists:
            os.makedirs(original_path+'test_ins_a')

        isExists=os.path.exists(original_path+'test_ins_b')
        if not isExists:
            os.makedirs(original_path+'test_ins_b')


        i=0
        ids_A,id_B=self.__load_class(name)
        #print(ids)
        #print(name[0])
        self.load__label(name[0],ids_A)
        self.load__label(name[1],id_B)
        for imagename_A in ids_A:
            imagename_B=id_B[i]
            if(i>=200):
                break
            elif i<150:
                save_path=self.rootpath+'file/'+str(name[0])+'/'+str(imagename_A)+'/'+ str(imagename_A)
                train_a=cv2.imread(save_path + 'com.jpg',1)
                train_a=cv2.resize(train_a,(256,256))
                cv2.imwrite(original_path+'train_A/'+str(i)+'.jpg',train_a)
                train_ins_a=cv2.imread(save_path+'com_label.jpg',1)
                train_ins_a=cv2.resize(train_ins_a,(64,64))
                cv2.imwrite(original_path+'train_ins_a/'+str(i)+'.jpg',train_ins_a)

                save_path=self.rootpath+'file/'+str(name[1])+'/'+str(imagename_B)+'/'+ str(imagename_B)
                train_b=cv2.imread(save_path + 'com.jpg',1)
                train_b=cv2.resize(train_b,(256,256))
                cv2.imwrite(original_path+'train_B/'+str(i)+'.jpg',train_b)
                train_ins_b=cv2.imread(save_path+'com_label.jpg',1)
                train_ins_b=cv2.resize(train_ins_b,(64,64))
                cv2.imwrite(original_path+'train_ins_b/'+str(i)+'.jpg',train_ins_b)
            else:
                save_path=self.rootpath+'file/'+str(name[0])+'/'+str(imagename_A)+'/'+ str(imagename_A)
                train_a=cv2.imread(save_path + 'com.jpg',1)
                train_a=cv2.resize(train_a,(256,256))
                cv2.imwrite(original_path+'test_A/'+str(i%150)+'.jpg',train_a)
                train_ins_a=cv2.imread(save_path+'com_label.jpg',1)
                train_ins_a=cv2.resize(train_ins_a,(64,64))
                cv2.imwrite(original_path+'test_ins_a/'+str(i%150)+'.jpg',train_ins_a)


                save_path=self.rootpath+'file/'+str(name[1])+'/'+str(imagename_B)+'/'+ str(imagename_B)
                train_b=cv2.imread(save_path + 'com.jpg',1)
                train_b=cv2.resize(train_b,(256,256))
                cv2.imwrite(original_path+'test_B/'+str(i%150)+'.jpg',train_b)
                train_ins_b=cv2.imread(save_path+'com_label.jpg',1)
                train_ins_b=cv2.resize(train_ins_b,(64,64))
                cv2.imwrite(original_path+'test_ins_b/'+str(i%150)+'.jpg',train_ins_b)
            i+=1
        a_identity=np.load(self.rootpath+'file/cat/final.npy')
        train_a_identity=a_identity[:150]
        np.save(original_path+'train_a_identity.npy',train_a_identity)
        test_a_identity=a_identity[150:200]
        np.save(original_path+'test_a_identity.npy',test_a_identity)
        b_identity=np.load(self.rootpath+'file/dog/final.npy')
        train_b_identity=b_identity[:150]
        np.save(original_path+'train_b_identity.npy',train_b_identity)
        test_b_identity=b_identity[150:200]
        np.save(original_path+'test_b_identity.npy',test_b_identity)
        #
        pad_matrix=np.load(original_path+'train_a_identity.npy')
        #print(pad_matrix.shape)
        #print("hello")
        for i in range(150):
            train_a=cv2.imread(original_path+'train_A/'+str(i)+'.jpg')
            identity_train_a=np.load(original_path+'train_a_identity.npy')
            identity_train_a=identity_train_a[i]
            shape=identity_train_a.shape
            ymin = int(pad_matrix[i][1] - 1)
            ymax = int(pad_matrix[i][3] - 1)
            xmin = int(pad_matrix[i][0] - 1)
            xmax = int(pad_matrix[i][2] - 1)
            train_a=cv2.resize(train_a,(256,256))
            ins_a=cv2.imread(original_path+'train_ins_a/'+str(i)+'.jpg')
            shape=train_a[ymin:ymax,xmin:xmax,:].shape
            ins_a=cv2.resize(ins_a,(shape[1],shape[0]))
            train_a[ymin:ymax,xmin:xmax,:]=ins_a
            cv2.imwrite(original_path+''+str(i)+'.jpg',train_a)

