
import numpy as np
import cv2
import NeuralNetworkSkeleton
import SIFT

# from SIFTfunc import *
level1_bounds=[(0,0),(640,480),(417,316),(417+80,316+50),(123,316),(123+80,316+50)]

game= [cv2.imread("images/Level1.png"),cv2.imread("images/Level2.png")]

finish_color=game[0][148,164]

loading= cv2.imread("images/loading.png")

win= cv2.imread("images/You win.png")
lose= cv2.imread("images/you lose.png")


# global frameCount
frameCount = 1


def detect_joints(img,level):
    bounds=[]
    if level==1:
        bounds=np.copy(level1_bounds)  

    points= NeuralNetworkSkeleton.SkeletonDetection(img)
    cond=True
    k=0
   
    for i in range(len(points)):
        if points[i] is None:
            cond = False
        else:
            cond=cond and points[i][0]>bounds[k][0]-50 and points[i][0]<bounds[k+1][0]+50 and points[i][1]>bounds[k][1]-50 and points[i][1]<bounds[k+1][1]+50
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


def ini(img,level):

    
    if level==1:
        START=(detect_skin(img,level1_bounds))
        i=0
        while i<len(level1_bounds)-1:
            cv2.rectangle(img,level1_bounds[i],level1_bounds[i+1],(0,0,255),3)
            i=i+2

        return START
    return False


def check_state(points,game):
    cond_win=True
    cond_loss=False
    for i in range(len(points)):
        cond_win=cond_win and not (False in (game[int(points[i][1]),int(points[i][0])]==finish_color))
        cond_loss= cond_loss or not (False in (game[int(points[i][1]),int(points[i][0])]==[0,0,0])) #black, out!!!
    return (cond_win, cond_loss)


#Open a simple image
cap = cv2.VideoCapture(0)

# cv2.setMouseCa/llback('image',draw_circle)
# frame=img
START=False
DETECTED=False
PLAYING =False
points=[]
level=1
while(True):


    ret,img = cap.read()
    img= cv2.flip(img,1)

    if START == False:
        START=ini(img,level)
        cv2.imshow("Game",img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif DETECTED==False:
        cv2.imshow("Game",loading)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        [DETECTED,points]= detect_joints(img,level)
        START=DETECTED#caLL detection
        PLAYING= DETECTED
        print("detect",DETECTED)
    elif (PLAYING==True):
        # print("before",points)
        points[1:3]= SIFT.track_it(img,[points[1],points[2]],frameCount)
        # print("after",points)
        # state = check_state(points,game[level-1])
        state=[False,False]
        
        if state[0]==True:
            PLAYING=False
            cv2.imshow("Game",win)
        elif state[1]==True:
            cv2.imshow("Game",lose)
        else:
            gamecpy=np.copy(game[level-1])
            
            for i in range(len(points)):
                if level==1 and i==0:
                    continue
                cv2.circle(gamecpy,(int(points[i][0]),int(points[i][1])),10,(0,0,255),thickness=-1,lineType=cv2.FILLED) 
                cv2.circle(img,(int(points[i][0]),int(points[i][1])),10,(0,0,255),thickness=-1,lineType=cv2.FILLED) 

            cv2.imshow("real",img)
            cv2.imshow("Game",gamecpy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, rele0ase the capture
cap.release()
cv2.destroyAllWindows() 

