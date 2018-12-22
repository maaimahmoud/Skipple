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
    for i in range(nPoints):
        if (i==0 or i == 1 or i==4 or i==7):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                points.append((int(x), int(y)))
            else :
                points.append(None)


    if (points[0] is None or points[1] is None):
        facePoint = None
    else:
        facePoint=(int((points[0][0]+points[1][0])/2),int((points[0][1]+points[1][1])/2))

    return [facePoint,points[3],points[2] ]
    