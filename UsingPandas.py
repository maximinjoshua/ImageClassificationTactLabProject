import cv2 as cv
import pandas as pd
import os
import numpy as np

os.chdir(r'D:\tacti and tact\temporary folder\uncutcomplete')
paths=[]
for filename in os.listdir("D:/tacti and tact/temporary folder/uncutcomplete"):
    paths.append(os.path.abspath(filename))

newpaths=[]
for names in paths:
    newpaths.append(names.replace('\\','/'))

image_data=[]
for paths in newpaths:
    image=cv.imread(paths)
    resized_image=image.resize((64,64))
    flattened_image=image.flatten()
    image_data.append(flattened_image)

os.chdir(r'D:\tacti and tact\temporary folder\cutcomplete')
paths1=[]
for filename in os.listdir("D:/tacti and tact/temporary folder/cutcomplete"):
    paths1.append(os.path.abspath(filename))

newpaths1=[]
for names in paths1:
    newpaths1.append(names.replace('\\','/'))

image_data1=[]
for paths in newpaths1:
    image=cv.imread(paths)
    resized_image=image.resize((64,64))
    flattened_image=image.flatten()
    image_data1.append(flattened_image)

image_data2=image_data+image_data1
dataframe=pd.DataFrame(image_data2)
labels1=np.zeros(len(image_data), dtype=np.int8)
labels2=np.ones(len(image_data1), dtype=np.int8)
labels=np.append(labels1,labels2)
dataframe['Labels']=labels

dataframe.to_csv(r"D:\tacti and tact\csvfiles\imagedata.csv", index=False)



    
    
