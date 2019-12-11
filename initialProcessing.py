
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import mrcnn
import scipy.misc
from PIL import Image

filedata=pd.read_csv('../../capstone_l/signDatabasePublicFramesOnly/allAnnotations.csv',sep=';') #path for the csv file with all annotations and the filepath
negativefilepath='../../capstone_l/signDatabasePublicFramesOnly/negatives/negativePics/'
negativeimagedir=[]
for i in os.listdir(negativefilepath):
    if i.endswith('.png'):
        negativeimagedir.append(os.path.join(negativefilepath,i))

imagesloaded=[]
annotationofimgs=[]
UCoordinate,LCoordinate=[],[]
for i in range(len(filedata)):
    img = cv2.imread('../../capstone_l/signDatabasePublicFramesOnly/'+filedata['Filename'][i],cv2.IMREAD_GRAYSCALE)
    imagesloaded.append(img)
    annotationofimgs.append(filedata['Annotation tag'][i])
    UCoordinate.append([filedata['Upper left corner X'][i],filedata['Upper left corner Y'][i]])
    LCoordinate.append([filedata['Lower right corner X'][i],filedata['Lower right corner Y'][i]])

negimagesloaded=[]
for i in range(len(negativeimagedir)):
    negimagesloaded.append(ocv.imread(negativeimagedir[i],ocv.IMREAD_GRAYSCALE))

uniqueshape=[]
for i in range(7855):
    uniqueshape.append(imagesloaded[i].shape)
for j in range(11634):
    uniqueshape.append(negimagesloaded[j].shape)
uniqueshape=list(set(uniqueshape))

reverse =  lambda x : (x[1],x[0])
for i in range(len(imagesloaded)):
    size=imagesloaded[i].shape
    defaultsize=reverse(min(uniqueshape))
    imagesloaded[i]=cv2.resize(imagesloaded[i],defaultsize)
    UCoordinate[i]=[int(UCoordinate[i][0]*defaultsize[0]/size[1]),int(UCoordinate[i][1]*defaultsize[1]/size[0])]
    LCoordinate[i]=[int(LCoordinate[i][0]*defaultsize[0]/size[1]),int(LCoordinate[i][1]*defaultsize[1]/size[0])]
for i in range(len(negimagesloaded)):
    size=negimagesloaded[i].shape
    negimagesloaded[i]=cv2.resize(negimagesloaded[i],defaultsize)
    
for i in range(len(imagesloaded)):
    temp=Image.fromarray(imagesloaded[i])
    temp.save("../rcnn/Mask_RCNN/training_images/{}.png".format(i))
for i in range(len(negimagesloaded)):
    temp=Image.fromarray(negimagesloaded[i])
    temp.save("../rcnn/Mask_RCNN/training_images/{}.png".format(len(imagesloaded)+i))

textfile={}
textfile['format']=[]
for i in range(len(filedata)):
    textfile['format'].append("training_images/{}.png,{},{},{},{},{}".format(i,UCoordinate[i][0],UCoordinate[i][1],LCoordinate[i][0],LCoordinate[i][1],filedata['Annotation tag'][i]))
textf=pd.DataFrame.from_dict(textfile)

textf.to_csv('../rcnn/Mask_RCNN/annotation.txt',header=None,index=None,sep=' ')
