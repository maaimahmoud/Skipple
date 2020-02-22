import numpy as np
from scipy.spatial.distance import euclidean
import cv2
import glob
import time

kpo=[]
kpo_des=[]

# No changes Here
def Initialization_point(kp,des,pointX,pointY):
    avgx=0
    avgy=0
    num=0
    kpo=[]
    kpo_des=[]
    
    for i in range(len(kp)):
            d=euclidean(kp[i].pt,[pointX,pointY])
            if d<15:
                kpo.append(kp[i])
                kpo_des.append(des[i])
                avgx+=kp[i].pt[0]
                avgy+=kp[i].pt[1]
                num+=1
    return [avgx,avgy,num,kpo,kpo_des]


#plenty of changes here 
def Track(kp,des,kpo,kpo_des):
    
    kpn_des=[]
    kpn=[]
    indices=[]
    dispx=0 # the displacement in x axis
    dispy=0 #  ~       ~        ~ y  ~
    mindiff=1e10 # The variable that holds the least descriptor difference 
    minidxdiff=-1 # The variable that holds the index of the old keypoint that has the least amount of difference occured 
    mininddiff=-1 # The variable that holds the index og the current keyopint that is the least one different in descriptor than others
    
    for i in range(len(kpo)):
        mindis=1e10
        minind=-1
        for j in range(len(kp)):
            d=euclidean(kpo_des[i],des[j])
            if d<mindis:
                mindis=d
                minind=j
        indices.append(minind)
        kpn.append(kp[minind])
        kpn_des.append(des[minind])
        if(mindiff>mindis): # Check to carry out the old keypoint and the current keypoint of the descriptor with least difference
            mindiff=mindis
            minidxdiff=i
            mininddiff=minind

    # setting displacement moved in both x,y

    if mininddiff !=-1 and minidxdiff !=-1:
        dispx=kp[mininddiff].pt[0]-kpo[minidxdiff].pt[0]
        dispy=kp[mininddiff].pt[1]-kpo[minidxdiff].pt[1]
    else:
        dispx = 0
        dispy = 0

    #These checks to prevent shaky vibrations through eliminating small displacements 
    if(np.abs(dispx)<0.7):
        dispx=0
    if(np.abs(dispy)<0.7):
        dispy=0
    
    return [kpn_des,kpn,dispx,dispy]


#small change just in the return 
def track_it(frame, points,f):  
    # print(points,f)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    
    # This check to prevent craching of the program through passing no keypoints to be tracked
    # if(len(kp)==0):
    #     return (-1,-1)
    
    cond=True
    # print ("SIFT",points)
    for i in range(len(points)):
        points[i]=SIFTfunc(frame,points[i],kp,des,i,f)
        if(points[i][1]==-1):
            cond=False

    # print("points in sift",points)
    
    return [points,cond]


#small change just in the return 
def SIFTfunc(frame,point, kps, dess, index,f):
    if f == 1:
        [point_avgx,point_avgy,point_num,kpoi,kpo_desi] = Initialization_point(kps,dess,point[0],point[1])
        if(point_num==0):
            # print("in point_num")
            return (-1,-1)
        point= ((point_avgx/point_num),(point_avgy/point_num))
        
        kpo.append(kpoi)
        kpo_des.append(kpo_desi)

        return point
    #----------------------------------------------------------------    Tracking     ----------------------------------------------------------------------#
    else:    
        allkp=np.copy(kps)
        alldes=np.copy(dess)
        kp=[]
        des=[]
        
        for i in range(len(allkp)):
            if (euclidean(allkp[i].pt,[point[0],point[1]])<60):
                kp.append(allkp[i])
                des.append(alldes[i])
       
        # This check to prevent crashing through looping in no keypoints
        if(len(kp)==0):
            # print("in len kp")
            return (-1,-1)       
        
        [point_kpn_des,point_kpn,point_dispx,point_dispy] = Track(kp,des,kpo[index],kpo_des[index])
        
        # # This check prevents abrupt sudden changes 
        # if(point_dispx>30 or point_dispy>30):     
        #     print("in disp size")
        #     return (-1,-1) 
        
        # Updating point to be drawn
        point_pointX=point[0]+point_dispx
        point_pointY=point[1]+point_dispy

        point_kpn=[] 
        point_kpn_des=[]
        
        for j in range(len(allkp)):
                di=euclidean([point_pointX,point_pointY],allkp[j].pt)
                if di<15:
                    point_kpn.append(allkp[j])
                    point_kpn_des.append(alldes[j])
                    # cv2.circle(frame,(int(allkp[j].pt[0]+0.5),int(allkp[j].pt[1]+0.5)),2,(0,255,255),thickness=-1,lineType=cv2.FILLED)
    
        kpo[index]=np.copy(point_kpn)
        kpo_des[index]= np.copy(point_kpn_des)

        return (point_pointX,point_pointY)

