import numpy as np
from scipy.spatial.distance import euclidean
import cv2
import glob
import time

global kpo,kpo_des
kpo=[]
kpo_des=[]

def Initialization_point(kp,des,pointX,pointY):
    avgx=0
    avgy=0
    num=0
    kpo=[]
    kpo_des=[]
    
    for i in range(len(kp)):
            d=euclidean(kp[i].pt,[pointX,pointY])
            if d<30:
                kpo.append(kp[i])
                kpo_des.append(des[i])
                avgx+=kp[i].pt[0]
                avgy+=kp[i].pt[1]
                num+=1
    return [avgx,avgy,num,kpo,kpo_des]



def Track(kp,des,kpo,kpo_des):
    
    kpn_des=[]
    kpn=[]
    indices=[]
    avgx=0
    avgy=0
    num=0
    
    for i in range(len(kpo)):
        mindis=1e10
        minind=-1
        for j in range(len(kp)):
            d=euclidean(kpo_des[i],des[j])
            if d<mindis:
                mindis=d
                minind=j
        if(minind not in indices):
            indices.append(minind)
            kpn.append(kp[minind])
            avgx+=kp[minind].pt[0]
            avgy+=kp[minind].pt[1]
            kpn_des.append(des[minind])
            num+=1
    return [kpn_des,kpn,avgx,avgy,num]


def track_it(frame, points,frameCount):  
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
       
    for i in range(len(points)):
        points[i]=SIFTfunc(frame,points[i],kp,des,i,frameCount)
    
    frameCount+=1
    return points
        


def SIFTfunc(frame,point, kps, dess, index,frameCount ):
    if frameCount == 1:
        [point_avgx,point_avgy,point_num,kpoi,kpo_desi] = Initialization_point(kps,dess,point[0],point[1])
        point= ((point_avgx/point_num),(point_avgy/point_num))
        kpo.append(kpoi)
        kpo_des.append(kpo_desi)

        return point
    #---------------------------------------------------------------- Tracking ----------------------------------------------------------------------#
    else:    
        allkp=kps
        alldes=dess
        kp=[]
        des=[]
        
        for i in range(len(allkp)):
            if (euclidean(allkp[i].pt,[point[0],point[1]])<30):
                kp.append(allkp[i])
                des.append(alldes[i])
        
        
        [point_kpn_des,point_kpn,point_avgx,point_avgy,point_num] = Track(kp,des,kpo[index],kpo_des[index])
        
        if not point_num==0:
            point_pointX=(point_avgx/point_num)
            point_pointY=(point_avgy/point_num)
        
        point_kpn=[] 
        point_kpn_des=[]
        
        for j in range(len(allkp)):
                di=euclidean([point_pointX,point_pointY],allkp[j].pt)
                if di<30:
                    point_kpn.append(allkp[j])
                    point_kpn_des.append(alldes[j])
    
        kpo[index]=np.copy(point_kpn)
        kpo_des[index]= np.copy(point_kpn_des)

        return (point_pointX,point_pointY)
        