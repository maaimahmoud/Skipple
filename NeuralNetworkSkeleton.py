import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# ## Specify the model to be used

protoFile = "NN/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "NN/pose_iter_160000.caffemodel"

global net 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def SkeletonDetection(frame):

    nPoints = 15

    inWidth = 368
    inHeight = 368

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []  
    auxpoints=[]
    for i in range(nPoints):
        if (i==0 or i == 1 or i==4 or i==7 or i==3 or i==6):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                if(i==3 or i==6):
                    auxpoints.append((int(x+0.5), int(y+0.5)))
                else:
                    points.append((int(x+0.5), int(y+0.5)))
            else :
                if(i==3 or i==6):
                    auxpoints.append(None)
                else:
                    points.append(None)

    # for (x,y) in points:
    #     cv2.circle(frame,(x,y),2,(0,0,255),thickness=-1,lineType=cv2.FILLED)

    if(auxpoints[0] is not None and auxpoints[1] is not None and points[2] is not None and points[3] is not None):
        diff1=(0.5*(points[2][0]-auxpoints[0][0]),0.5*(points[2][1]-auxpoints[0][1]))
        points[2]= ( int( points[2][0]+diff1[0]+0.5), int( points[2][1]+diff1[1]+0.5))
        diff2=(0.5*(points[3][0]-auxpoints[1][0]),0.5*(points[3][1]-auxpoints[1][1]))
        points[3]= ( int( points[3][0]+diff2[0]+0.5), int( points[3][1]+diff2[1]+0.5))


#    print auxilarty points of elbow
    # cv2.circle(frame,(auxpoints[0][0],auxpoints[0][1]),2,(0,255,255),thickness=-1,lineType=cv2.FILLED)
    # cv2.circle(frame,(auxpoints[1][0],auxpoints[1][1]),2,(0,255,255),thickness=-1,lineType=cv2.FILLED)

#
    # cv2.circle(frame,(points[3][0],points[3][1]),2,(0,255,0),thickness=-1,lineType=cv2.FILLED)
    # cv2.circle(frame,(points[2][0],points[2][1]),2,(0,255,0),thickness=-1,lineType=cv2.FILLED)

    # while(1):
    #     cv2.imshow("joints",frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    if (points[0] is None or points[1] is None):
        facePoint = None
    else:
        facePoint=(int(((points[0][0]+points[1][0])/2)+0.5),int(((points[0][1]+points[1][1])/2)+0.5))

    return [facePoint,points[3],points[2] ]
    