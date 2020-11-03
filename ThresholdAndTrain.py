import cv2
import numpy as np
import os

os.chdir(r'D:\tacti and tact\temporary folder\uncut')
paths=[]
for path0 in os.listdir("D:/tacti and tact/temporary folder/uncut"):
    paths.append(os.path.abspath(path0))

newpaths=[]
for names in paths:
    newpaths.append(names.replace('\\','/'))

os.chdir(r"D:\tacti and tact\threshimage\uncut")
n=0
for i in range(0,len(newpaths)):
    image=cv2.imread(newpaths[i])
    conversion=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    retval,threshold=cv2.threshold(conversion,240,255,cv2.THRESH_BINARY)
    filename="thresh"+str(n)+".jpg"
    cv2.imwrite(filename,threshold)
    n=n+1



    


    
    
