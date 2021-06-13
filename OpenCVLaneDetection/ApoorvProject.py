import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
cap = cv.VideoCapture("/home/apoorv/Videos/lane_vgt.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    LOWER_WHITE=np.array([40,42,180])
    HEIGHER_WHITE=np.array([73,140,236])
    mask = cv.inRange(hsv, LOWER_WHITE, HEIGHER_WHITE)
    res = cv.bitwise_and(frame,frame, mask= mask)
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    #equ = cv.equalizeHist(gray)
    _ , thr=cv.threshold(gray,170,255,cv.THRESH_BINARY)
    #plt.imshow(frame)
    #plt.show()
    #cv.imshow('thr', thr)
    cv.imshow('Threshold',res)
    blur = cv.GaussianBlur(thr,(5,5),0)
    edges = cv.Canny(blur,150,300)
    #cv.imshow('edges',edges)
    lines = cv.HoughLinesP(
        edges,
        rho =1.0,
        theta =np.pi/180,
        threshold =20,
        minLineLength =30,
        maxLineGap =15
    )
    line_img =np.zeros((frame.shape[0], frame.shape[1],3), dtype=np.uint8)
    line_color = [255,0,0]
    line_thickness = 1
    dot_color =[255,0,0]
    dot_size =3
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_img,(x1,y1),(x2,y2),line_color,line_thickness)
            cv.circle(line_img,(x1,y1),dot_size,dot_color,-1)
            cv.circle(line_img,(x2,y2),dot_size,dot_color,-1)
    end=cv.addWeighted(frame,0.8,line_img,1.0,0.0)
    cv.imshow('end',end)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()