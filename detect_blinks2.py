#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm
from sklearn.externals import joblib
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import time
from skimage.feature import hog
import joblib


VECTOR_SIZE = 3
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#pwd = os.getcwd()
#model_path = os.path.join(pwd, 'model')
shape_detector_path = 'shape_predictor_68_face_landmarks.dat'


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

# 导入模型
clf = joblib.load("ear_svm.m")

EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 当EAR小于阈值时，接连多少帧一定发生眨眼动作



frame_counter = 0
blink_counter = 0
ear_vector = []
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)
while(1):
    if fileStream and not vs.more():
        break
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    #print(len(rects))
    for rect in rects:
        print('-'*20)
        print(rect)
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
        # points = shape.parts()
        leftEye = points[lStart:lEnd]
        rightEye = points[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        #print(leftEyeHull)
        rightEyeHull = cv2.convexHull(rightEye)
        xmaxl=max(leftEyeHull, key=lambda x: x[0][0])[0][0]
        xminl=min(leftEyeHull, key=lambda x: x[0][0])[0][0]
        yminl = min(leftEyeHull, key=lambda x: x[0][1])[0][1]
        ymaxl = max(leftEyeHull, key=lambda x: x[0][1])[0][1]
        xmaxr=max(rightEyeHull, key=lambda x: x[0][0])[0][0]
        xminr=min(rightEyeHull, key=lambda x: x[0][0])[0][0]
        yminr = min(rightEyeHull, key=lambda x: x[0][1])[0][1]
        ymaxr = max(rightEyeHull, key=lambda x: x[0][1])[0][1]
        #print(xminl)
        #print(yminl)
        leftPart=cv2.resize(gray[yminl-10:ymaxl+10,xminl-10:xmaxl+10].copy(),(50,20), interpolation=cv2.INTER_AREA)
        rightPart=cv2.resize(gray[yminr-10:ymaxr+10,xminr-10:xmaxr+10].copy(),(50,20), interpolation=cv2.INTER_AREA)

        cv2.imwrite('left.jpg',leftPart)
        cv2.imwrite('right.jpg',rightPart)
        test =[]

        print(leftPart.shape)
        fd,hog_image = hog(leftPart, orientations=8, pixels_per_cell=(5,5),cells_per_block=(1, 1),block_norm= 'L2',visualise=True)
        test.append(fd)
        result = clf.predict(test)
        print('result of leftEye is ')
        print(result)
        if result ==0:

            cv2.putText(frame, "leftEye:close", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "leftEye:Open", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        ret, ear_vector = queue_in(ear_vector, ear)
        if(len(ear_vector) == VECTOR_SIZE):
            #print(ear_vector)
            #input_vector = []
            #input_vector.append(ear_vector)
            #print(res)

            if result == 1:
                frame_counter += 1
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0
                break

        cv2.putText(frame, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #cv2.putText(frame, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
