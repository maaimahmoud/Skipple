import numpy as np
from scipy.spatial.distance import euclidean
import cv2
import glob
import time

frameCount=1 #Not sure about the global thing 
kpo=[]
kpo_des=[]

# Initializing the white point to be tracked
x=400  
y=300

# The last point that are passed to th function in order to modify it with the calculated displacement values
lastx=400
lasty=300


# No changes Here
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
    dispx=kp[mininddiff].pt[0]-kpo[minidxdiff].pt[0]
    dispy=kp[mininddiff].pt[1]-kpo[minidxdiff].pt[1]
    
    #These checks to prevent shaky vibrations through eliminating small displacements 
    if(np.abs(dispx)<0.7):
        dispx=0
    if(np.abs(dispy)<0.7):
        dispy=0
    
    return [kpn_des,kpn,dispx,dispy]


#small change just in the return 
def track_it(frame, points,f):  
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    
    # This check to prevent craching of the program through passing no keypoints to be tracked
    if(len(kp)==0):
        return (-1,-1)

    for i in range(len(points)):
        points[i]=SIFTfunc(frame,points[i],kp,des,i,f)
    
    return points


#small change just in the return 
def SIFTfunc(frame,point, kps, dess, index,f ):
    if f == 1:
        [point_avgx,point_avgy,point_num,kpoi,kpo_desi] = Initialization_point(kps,dess,point[0],point[1])
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
            if (euclidean(allkp[i].pt,[point[0],point[1]])<50):
                kp.append(allkp[i])
                des.append(alldes[i])
       
        # This check to prevent crashing through looping in no keypoints
        if(len(kp)==0):
            return (-1,-1)       
        
        [point_kpn_des,point_kpn,point_dispx,point_dispy] = Track(kp,des,kpo[index],kpo_des[index])
        
        # This check prevents abrupt sudden changes 
        if(point_dispx>30 or point_dispy>30):
            return (-1,-1) 
        
        # Updating point to be drawn
        point_pointX=lastx+point_dispx
        point_pointY=lasty+point_dispy

        point_kpn=[] 
        point_kpn_des=[]
        
        for j in range(len(allkp)):
                di=euclidean([point_pointX,point_pointY],allkp[j].pt)
                if di<30:
                    point_kpn.append(allkp[j])
                    point_kpn_des.append(alldes[j])
                    cv2.circle(frame,(int(x+0.5),int(y+0.5)),2,(0,255,255),thickness=-1,lineType=cv2.FILLED)
    
        kpo[index]=np.copy(point_kpn)
        kpo_des[index]= np.copy(point_kpn_des)

        return (point_pointX,point_pointY)


#****************************************************************      MAIN       **********************************************************************************

cap = cv2.VideoCapture(0)
start=False
while(True):
    ret,img = cap.read()
    img= cv2.flip(img,1)
    
    #White point 
    if(start==False):
        cv2.circle(img,(int (x),int(y)),3,(255,255,255),thickness=-1,lineType=cv2.FILLED)
        frameCount+=1
        if(frameCount>100):
            frameCount=1
            start=True
        cv2.imshow("real",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    [(x,y)]=track_it(img,[(lastx,lasty)],frameCount)
    # For any reason -1,-1 means a false tracking attempt
    if(x==-1 or y==-1):
        x=lastx
        y=lasty
    
    frameCount+=1 # I incremented the frameCount here because the global variable thing was not working for some reason
    cv2.circle(img,(int(x+0.5),int(y+0.5)),10,(0,0,255),thickness=-1,lineType=cv2.FILLED)
    lastx=x
    lasty=y
    prev_img = np.copy(img)
    cv2.imshow("real",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, rele0ase the capture
cap.release()
cv2.destroyAllWindows() 