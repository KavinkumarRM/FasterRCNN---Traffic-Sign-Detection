import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import random

#### AUGMENTATION#######


data=pd.read_csv('../../capstone_l/signDatabasePublicFramesOnly/allAnnotations.csv',sep=';')
data['Annotation tag'].value_counts().plot(kind='bar') #bar graph for distribution

class augment_data: #augmenting the minipics out based on certain criteria
    def __init__(self,master_data,augment=False,filepath='./',load_img=False):
        if type(master_data)==pd.core.frame.DataFrame:
            self.m_data=master_data
        else:
            raise ValueError('Not a Dataframe')
        if augment==True:
            self.values_to_resample,self.pics_to_consider=self.resample_augment()
        else:
            self.values_to_resample=self.resample_without_augment()
            self.pics_to_consider=None
        self.load_img=load_img
        self.shape=(480, 640)
        self.filepath=filepath
        self.imagesloaded,self.negimagesloaded,self.annotationofimgs,self.UCoordinate,self.LCoordinate,self.negativeimagedir=self.read_pics()
        self.mini_pics=self.get_minipics()
        
    def valcount(self):
        valuescount=self.m_data['Annotation tag'].value_counts()
        return(valuescount)
        
    def resample_augment(self):
        x=self.valcount().to_dict()
        labels=list(x.keys())
        values=list(x.values())
        valuesres={}
        picstobeextracted={}
        for j,i in x.items():
            if i>=500:
                valuesres[j]=900
            elif i>=250:
                valuesres[j]=500
            elif i>=130:
                valuesres[j]=250
            elif i>=70:
                valuesres[j]=150
            elif i>=30:
                valuesres[j]=130
            elif i<30:
                valuesres[j]=100
            if i>=valuesres[j]:
                picstobeextracted[j]=False
            else:
                picstobeextracted[j]=True
        return(valuesres,picstobeextracted)

    def resample_without_augment(self):
        x=valcount().to_dict()
        labels=list(x.keys())
        values=list(x.values())
        valuesres={}
        for j,i in x.items():
            if i>=500:
                valuesres[j]=900
            if i>=250:
                valuesres[j]=500
            if i>=130:
                valuesres[j]=250
            if i>=70:
                valuesres[j]=150
            if i>=30:
                valuesres[j]=130
            if i<=30:
                valuesres[j]=100
        return(valuesres)

    def read_pics(self):
        negativeimagedir=[]
        for i in os.listdir(self.filepath+'negatives/negativePics/'):
            if i.endswith('.png'):
                negativeimagedir.append(os.path.join(self.filepath+'negatives/negativePics/',i))
        negimagesloaded=[]
        if self.load_img==True:
            for i in range(len(negativeimagedir)):
                negimagesloaded.append(cv2.imread(negativeimagedir[i],cv2.IMREAD_GRAYSCALE))
        imagesloaded=[]
        annotationofimgs=[]
        UCoordinate,LCoordinate=[],[]
        for i in range(len(self.m_data)):
            if self.load_img==True:
                img = cv2.imread(self.filepath+self.m_data['Filename'][i],cv2.IMREAD_GRAYSCALE)
                imagesloaded.append(img)
            annotationofimgs.append(self.m_data['Annotation tag'][i])
            UCoordinate.append([self.m_data['Upper left corner X'][i],self.m_data['Upper left corner Y'][i]])
            LCoordinate.append([self.m_data['Lower right corner X'][i],self.m_data['Lower right corner Y'][i]])
        return(imagesloaded,negimagesloaded,annotationofimgs,UCoordinate,LCoordinate,negativeimagedir)

    def get_area_index(self):
        area_index={}
        area={}
        for i in range(len(self.annotationofimgs)):
            if self.pics_to_consider[self.annotationofimgs[i]]==True:
                if self.annotationofimgs[i] not in area.keys():
                    area[self.annotationofimgs[i]]=0
                temp=(self.LCoordinate[i][0]-self.UCoordinate[i][0])*(self.LCoordinate[i][1]-self.UCoordinate[i][1])
                if temp>area[self.annotationofimgs[i]]:
                    area[self.annotationofimgs[i]]=temp
                    area_index[self.annotationofimgs[i]]=i
        return (area_index)
    def resize(self,img,j):
        reverse =  lambda x : (x[1],x[0])
        size=img.shape
        defaultsize=reverse(self.shape)
        img=cv2.resize(img,defaultsize)
        UC=[int(self.UCoordinate[j][0]*defaultsize[0]/size[1]),int(self.UCoordinate[j][1]*defaultsize[1]/size[0])]
        LC=[int(self.LCoordinate[j][0]*defaultsize[0]/size[1]),int(self.LCoordinate[j][1]*defaultsize[1]/size[0])]
        return(img,UC,LC)
    def get_minipics(self):
        area_index=self.get_area_index()
        minipics={}
        for i,j in area_index.items():
            temp_img_dir=self.filepath+self.m_data['Filename'][j]
            img=cv2.imread(temp_img_dir,cv2.IMREAD_GRAYSCALE)
            img,UC,LC=self.resize(img,j)
            mini_img=img[UC[1]:LC[1],UC[0]:LC[0]]
            minipics[i]=mini_img
        return(minipics)

aug_master_data=augment_data(data,augment=True,filepath='../../capstone_l/signDatabasePublicFramesOnly/',load_img=False)

class morph:
    def __init__(self,augmentdata_obj):
        self.obj=augmentdata_obj
        self.number_upsample=self.numbers()
        self.up_img,self.up_LC,self.up_UC=self.morph()
    def numbers(self):
        number_upsample={}
        for i,j in self.obj.pics_to_consider.items():
            if j==True:
                number_upsample[i]=int(self.obj.values_to_resample[i])-int(self.obj.valcount()[i])
            else:
                number_upsample[i]=0
        return(number_upsample)
    def get_next_RN(self):
        k=random.random()
        l=random.random()
        return(k,l)
    def unique_shape(self):
        uniqueshape=[]
        for i in range(7855):
            uniqueshape.append(self.obj.imagesloaded[i].shape)
        for j in range(11634):
            uniqueshape.append(self.obj.negimagesloaded[j].shape)
        uniqueshape=list(set(uniqueshape))
        return(uniqueshape)
    def resize_coordinates(self):
        self.uniqueshape=self.unique_shape()
        reverse =  lambda x : (x[1],x[0])
        imagesloaded=self.obj.imagesloaded
        negimagesloaded=self.obj.negimagesloaded
        UCoordinate=self.obj.UCoordinate
        LCoordinate=self.obj.LCoordinate
        for i in range(len(imagesloaded)):
            size=imagesloaded[i].shape
            defaultsize=reverse(min(self.uniqueshape))
            imagesloaded[i]=cv2.resize(imagesloaded[i],defaultsize)
            UCoordinate[i]=[int(UCoordinate[i][0]*defaultsize[0]/size[1]),int(UCoordinate[i][1]*defaultsize[1]/size[0])]
            LCoordinate[i]=[int(LCoordinate[i][0]*defaultsize[0]/size[1]),int(LCoordinate[i][1]*defaultsize[1]/size[0])]
        for i in range(len(negimagesloaded)):
            size=negimagesloaded[i].shape
            negimagesloaded[i]=cv2.resize(negimagesloaded[i],defaultsize)
        self.negimg=negimagesloaded
        self.img=imagesloaded
        self.UCoordinate=UCoordinate
        self.LCoordinate=LCoordinate
    def resize_minipic(self,img,UC,LC):
        shape=((LC[0]-UC[0]),(LC[1]-UC[1]))
        resized=cv2.resize(img,shape)
        return(resized)
    def morph(self):
        self.resize_coordinates()
        upsampled_img={}
        upsampled_UC={}
        upsampled_LC={}
        for i,j in self.obj.pics_to_consider.items():
            upsampled_img[i]=[]
            upsampled_LC[i]=[]
            upsampled_UC[i]=[]
            if j:
                for k in range(self.number_upsample[i]):
                    r1,r2=self.get_next_RN()
                    bottom_img=self.img[int(r2*len(self.img))]
                    top_img=self.obj.mini_pics[i]
                    UC=self.UCoordinate[int(r2*len(self.UCoordinate))]
                    LC=self.LCoordinate[int(r2*len(self.LCoordinate))]
                    top_resized=self.resize_minipic(top_img,UC,LC)
                    bottom_img[UC[1]:LC[1],UC[0]:LC[0]]=top_resized
                    upsampled_img[i].append(bottom_img)
                    upsampled_LC[i].append(LC)
                    upsampled_UC[i].append(UC)
        return(upsampled_img,upsampled_LC,upsampled_UC)

final_data=morph(aug_master_data)
imgs=0
for i in final_data.up_img.keys():
    imgs=imgs+len(final_data.up_img[i])
print(imgs+len(final_data.img))

textfile={}
textfile['format']=[]
for i in range(len(final_data.img)):
    textfile['format'].append("training_images/{}.png,{},{},{},{},{}".format(i,final_data.UCoordinate[i][0],final_data.UCoordinate[i][1],final_data.LCoordinate[i][0],final_data.LCoordinate[i][1],data['Annotation tag'][i]))
"""co=len(final_data.img)
p=0
for j in final_data.up_img.keys():
    for k in range(len(final_data.up_img[j])):
        textfile['format'].append("training_images/{}.png,{},{},{},{},{}".format(co+p,final_data.up_UC[j][k][0],final_data.up_UC[j][k][1],final_data.up_LC[j][k][0],final_data.up_LC[j][k][1],j))
        p=p+1"""
        
upsamplewithoutaug=pd.DataFrame.from_dict(textfile)
upsamplewithoutaug['annotation']=data['Annotation tag']
t=upsamplewithoutaug[upsamplewithoutaug['annotation']=='stop'].reset_index()
def upsample(x):
    temp_text={'format':[]}
    set_of_ann=set(x['annotation'])
    picstocons=aug_master_data.pics_to_consider
    valtosamp=aug_master_data.values_to_resample
    for i in set_of_ann:
        temp=x[x['annotation']==i].reset_index()
        if picstocons[i]==True:
            j=0
            while j<valtosamp[i]:
                try:
                    temp_text['format'].append(temp['format'][j])
                except:
                    temp_text['format'].append(temp['format'][int(len(temp)*random.random())])
                j=j+1
        else:
            for j in range(len(temp)):
                if j<valtosamp[i]:
                    temp_text['format'].append(temp['format'][j])
    return(temp_text)
text_file=upsample(upsamplewithoutaug)
t_f=pd.DataFrame.from_dict(text_file)
t_f.to_csv('../annotation_resampled.txt',header=None,index=None,sep=' ')

k=0
for i in range(len(final_data.img)):
    temp=Image.fromarray(final_data.img[i])
    temp.save("../training_images/{}.png".format(i))
    k=k+1
p=0
for j in final_data.up_img.keys():
    for k in range(len(final_data.up_img[j])):
        temp=Image.fromarray(final_data.up_img[j][k])
        temp.save("../training_images/{}.png".format(co+p))
        p=p+1
        k=k+1
textf=pd.DataFrame.from_dict(textfile)
textf.to_csv('../annotation_upsampled.txt',header=None,index=None,sep=' ')






