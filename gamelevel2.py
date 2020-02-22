
import numpy as np
import cv2
import NeuralNetworkSkeleton
import SIFTfunc

import time

# from SIFTfunc import *
level1_bounds=[(0,0),(640,480),(417,316),(417+80,316+50),(123,316),(123+80,316+50)]
level2_bounds=[(265,177),(347,251),(482,265),(575,311),(48,272),(116,318)]


game= [cv2.imread("images/Level1.png"),cv2.imread("images/Level2.png")]

finish_color=game[0][148,164]

loading= cv2.imread("images/loading.png")

win= cv2.imread("images/You win.png")
lose= cv2.imread("images/you lose.png")




def detect_joints(img,level):
    bounds=[]
    if level==1:
        bounds=np.copy(level1_bounds)  
    elif level == 2:
        bounds=np.copy(level2_bounds)

    points= NeuralNetworkSkeleton.SkeletonDetection(img)
    cond=True
    k=0
   
    for i in range(len(points)):
        if points[i] is None:
            cond = False
        else:
            cond=cond and points[i][0]>bounds[k][0]-0 and points[i][0]<bounds[k+1][0]+0 and points[i][1]>bounds[k][1]-0 and points[i][1]<bounds[k+1][1]+0
        k+=2

    # jointsFrame = np.copy(img)
    # if cond == True:
    #     for i in range(len(points)):
    #         cv2.circle(jointsFrame,(int(points[i][0]),int(points[i][1])),2,(255,0,255),2)
    #     cv2.imshow("joints",jointsFrame)

    return [cond,points]


# def print_coord(event,x,y,flags,param):
#     # global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print(x,y)
#         # mouseX,mouseY = x/,y

def detect_skin(img,bounds):
    
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    cond=True

    i=0
    while i<len(bounds)-1:
        image = np.copy(global_result[bounds[i][1]:bounds[i+1][1],bounds[i][0]:bounds[i+1][0]])
        cond =cond and ((image == 0).sum() != 0)
        i=i+2

        # image2= np.copy(global_result[bounds[2][1]:bounds[3][1],bounds[2][0]:bounds[3][0]])

    return cond

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('face_haar.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    away=False

    if len(faces)>0:
        (x,y,w,h) = faces[0]
        color=(255,0,0)

        if(w<70):
            color=(0,0,255)
            away=True

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        # print("face size ", w)

    return away


    

def ini(img,level):

    
    if level==1:
        START=detect_face(img) and (detect_skin(img,level1_bounds)) 
        i=2 
        while i<len(level1_bounds)-1:
            cv2.rectangle(img,level1_bounds[i],level1_bounds[i+1],(0,0,255),3)
            i=i+2

        return START

    if level==2:
        START=detect_face(img) and (detect_skin(img,level2_bounds)) 
        i=0
        while i<len(level2_bounds)-1:
            cv2.rectangle(img,level2_bounds[i],level2_bounds[i+1],(0,0,255),3)
            i=i+2

        return START


    return False


def check_state(points,game):
    cond_win=True
    cond_loss=False

    for i in range(len(points)):
        if level==1 and i!=0:
            cond_win=cond_win and not (False in (game[int(points[i][1]),int(points[i][0])]==finish_color))
            cond_loss= cond_loss or not (False in (game[int(points[i][1]),int(points[i][0])]==[0,0,0])) #black, out!!!
        elif level==2:
            cond_win=cond_win and not (False in (game[int(points[i][1]),int(points[i][0])]==finish_color))
            cond_loss= cond_loss or not (False in (game[int(points[i][1]),int(points[i][0])]==[0,0,0])) #black, out!!!
    return (cond_win, cond_loss)


#Open a simple image
cap = cv2.VideoCapture(0)

# cv2.SetWindowProperty("Game", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

exitGame = False
levelEnded = False

while (not exitGame or levelEnded):
    kpo=[]
    kpo_des=[]
    
    levelEnded = False
    exitGame = False

    cv2.namedWindow("Game",cv2.WINDOW_NORMAL)
    windowWidth = int(2*640)
    windowHeight = int(2*480)
    cv2.resizeWindow("Game", (int(windowWidth),int(windowHeight)))

    # cv2.setMouseCa/llback('image',draw_circle)
    # frame=img
    START=False
    DETECTED=False
    PLAYING =False
    points=[]

    # global frameCount
    frameCount = 1

    # __,img = cap.read()
    level=1

    wait_2=False

    start_wait=time.time()

    print("reseting")
    # Select level
    while (True):

        
        
        imgLevel = np.zeros((windowWidth,windowHeight))
    
        font                   = cv2.FONT_HERSHEY_PLAIN 
        bottomLeftCornerOfText = (100,290)
        fontScale              = 6
        fontColor              = (100,0,100)
        lineType               = 2

        cv2.putText(imgLevel,'SKIPPLE!', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)


        font                   = cv2.FONT_HERSHEY_PLAIN
        fontColor              = (255,0,0)
        lineType               = 2
        
        fontScale= 4
        bottomLeftCornerOfText = (300,600)
        cv2.putText(imgLevel,'Level 1', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        bottomLeftCornerOfText = (300,800)
        cv2.putText(imgLevel,'Level 2', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        bottomLeftCornerOfText = (300,1000)
        cv2.putText(imgLevel,'Level 3', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)


        if level == 1:
            bottomLeftCornerOfText = (100,600)
        elif level == 2:
            bottomLeftCornerOfText = (100,800)
        elif level == 3:
            bottomLeftCornerOfText = (100,1000)

        cv2.putText(imgLevel,"*", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

        cv2.imshow("Game",imgLevel)

        keyValue = cv2.waitKey(0)

        if keyValue & 0xFF == ord('q'):
            exitGame = True
            break

        if keyValue & 0xFF == ord('w'):
            level =max(level-1,1)
            # print("new level ",level)

        
        if keyValue & 0xFF == ord('s'):
            level =min(level+1,3)
            # print("new level ",level)
   
        if keyValue & 0xFF == ord('a'):
            break




    
    
    # keyValue = cv2.waitKey(1)

    
    while(not exitGame and not levelEnded):

        print("frameCount = ",frameCount)

        keyValue = cv2.waitKey(1)


        ret,img = cap.read()

        img= cv2.flip(img,1)

        if START == False:
            if((time.time()-start_wait)>2):
                START=ini(img,level)
                wait_2=False
            else:
                ini(img,level)

            cv2.imshow("Game",img)
            if keyValue & 0xFF == ord('q'):
                exitGame = True
        
        elif DETECTED==False:
            cv2.imshow("Game",loading)
            if keyValue & 0xFF == ord('q'):
                exitGame = True
                break
            [DETECTED,points]= detect_joints(img,level)
            # print(points)
            START=DETECTED#caLL detection
            PLAYING= DETECTED
            wait_2=not DETECTED
            start_wait= time.time()
            # print("detect",DETECTED)
        elif (PLAYING==True):
            # print("before sift",points)
            if level == 1:
                [my_points,yay]= SIFTfunc.track_it(img,[points[1],points[2]],frameCount)
            elif level == 2:
                [my_points,yay]= SIFTfunc.track_it(img,points,frameCount)
            # print("after sift",my_points)
            
            if yay and not my_points == [(-1,-1),(-1,-1)]:
                frameCount+=1
                if level == 1:
                    points[1:3]=my_points
                elif level == 2:
                    points[0:3]=my_points

            # print("after",points)
            state = check_state(points,game[level-1])
            # state=[False,False]
            
            if state[0]==True:
                PLAYING=False
                cv2.imshow("Game",win)
                levelEnded = True

            elif state[1]==True:
                cv2.imshow("Game",lose)
                levelEnded = True

            else:
                gamecpy=np.copy(game[level-1])
                for i in range(len(points)):
                    if level==1 and i==0:
                        continue
                    cv2.circle(gamecpy,(int(points[i][0]),int(points[i][1])),10,(0,0,255),thickness=-1,lineType=cv2.FILLED) 
                    cv2.circle(img,(int(points[i][0]),int(points[i][1])),10,(0,0,255),thickness=-1,lineType=cv2.FILLED) 

                cv2.imshow("Real",img)
                cv2.imshow("Game",gamecpy)

            if keyValue & 0xFF == ord('q'):
                exitGame = True
                break

        
        keyValue = cv2.waitKey(1)

            
        if levelEnded == True:
            keyValue = cv2.waitKey(0)
            # if keyValue & 0xFF == ord('d'):
            #     levelEnded = False
            break

            
            


        

        
    
    
# When everything done, rele0ase the capture
cap.release()
cv2.destroyAllWindows() 
