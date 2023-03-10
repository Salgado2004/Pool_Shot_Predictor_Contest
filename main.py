import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
        
def imgProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 2)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    return imgCanny

def findTaco(img):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([82,17,182])
    upper = np.array([99,88,255])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgFiltered = cv2.bitwise_and(img,img,mask=mask)
    imgProcessed = imgProcessing(imgFiltered)
    contours, hierarchy = cv2.findContours(imgProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 35 and area < 50:
            cv2.drawContours(imgFiltered, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor > 8:
                cv2.rectangle(imgFiltered, (x, y), (x+w, y+h), (255,255,255), 2)
                cv2.putText(imgCropped, "Taco", (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0,0,0), 2)

def findCueBall(img):
    croppedImg = img[51:349, 31:510]
    imgHSV = cv2.cvtColor(croppedImg,cv2.COLOR_BGR2HSV)
    lower = np.array([40,9,107])
    upper = np.array([72,96,210])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgFiltered = cv2.bitwise_and(croppedImg,croppedImg,mask=mask)
    imgProcessed = imgProcessing(imgFiltered)
    contours, hierarchy = cv2.findContours(imgProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 200 and area < 300:
            cv2.drawContours(imgFiltered, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor >= 8:
                cv2.putText(imgCropped, "Bola branca", (x+(w//2)+40, y+(h//2)+61), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
    return imgFiltered

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("resources/shots.mp4")
while True:
    success, frame = cap.read()
    imgRaw = cv2.resize(frame, (frameWidth, frameHeight))
    imgCropped = imgRaw[10:400,50:591]
    cv2.imwrite("imgColors.png", imgCropped)
    findTaco(imgCropped)
    cueBall = findCueBall(imgCropped)
    finalImg = stackImages(1, [imgRaw, cueBall])
    cv2.imshow("Result", finalImg)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break