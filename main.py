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

def colorFilter(img, lower, upper):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV,lower,upper)
    imgFiltered = cv2.bitwise_and(img,img,mask=mask)
    return imgFiltered
        
def imgProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 2)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    return imgCanny

def findTaco(img):
    lower = np.array([79,36,168])
    upper = np.array([98,255,217])
    imgFiltered = colorFilter(img, lower, upper)
    imgProcessed = imgProcessing(imgFiltered)
    contours, hierarchy = cv2.findContours(imgProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 15 and area < 65:
            cv2.drawContours(imgFiltered, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(imgFiltered, f"{w}, {h}" ,(x+w//2, y+h//2-10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(imgFiltered, f"{area}" ,(x+w//2, y+h//2+15),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 1)
            if objCor > 2:
                if w < 15 and w > 6 and h < 15 and h > 6:
                    cv2.rectangle(imgFiltered, (x, y), (x+w, y+h), (255,255,255), 2)
                    cv2.putText(imgCropped, "Taco", (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (0,0,0), 1)
                    return [x, y, w, h]
    #return imgFiltered

def findCueBall(img):
    croppedImg = img[56:383, 38:757]
    lower = np.array([40,9,107])
    upper = np.array([72,96,210])
    imgFiltered = colorFilter(croppedImg, lower, upper)
    imgProcessed = imgProcessing(imgFiltered)
    contours, hierarchy = cv2.findContours(imgProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv2.contourArea(i)
        if area > 400 and area < 500:
            cv2.drawContours(imgFiltered, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor >= 8:
                cv2.putText(imgCropped, "Bola branca", (x+(w//2)+40, y+(h//2)+61), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255,255,255), 1)
                return [x+31, y+51, w, h]

def findColoredBalls(img):
    kernel = np.ones((5,5),np.uint8)
    lowerValues = [[13,133,132], [74, 33, 71], [61, 36, 70], [82, 71, 72], [0, 98, 70], [61, 36, 70], [120,53,116], [0, 0, 0]]
    upperValues = [[66,255,255], [123, 255, 255], [79, 232, 255], [125, 255, 255], [17, 255, 255], [79, 232, 255], [179,255,255], [179, 255, 255]]
    filteredImgs = []
    for x in range(8):
        lower = np.array(lowerValues[x])
        upper = np.array(upperValues[x])
        filteredImgs.append(colorFilter(img, lower, upper))
    for f_img in filteredImgs:
        imgProcessed = imgProcessing(f_img)
        ballCount = 0
        foundBalls = []
        dial = cv2.dilate(imgProcessed, kernel, iterations=1)
        thres = cv2.erode(dial, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50 and area < 520:
                cv2.drawContours(f_img, i, -1, (172, 0, 196), 1)
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02*peri, True)
                objCor = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                if objCor > 7 and objCor < 14:
                    #cv2.putText(f_img, f"{w}", (x, y-15), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 1)
                    #cv2.putText(f_img, f"{h}", (x+w, y+(h//2)+15), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 1)
                    if h > 15 and h < 37 and w > 15 and w < 37 and (w-h) > -5.1 and (w-h) < 5.1:
                        cv2.circle(f_img, (x+w//2,y+h//2), (w//2), (255,255,0), 2)
                        ballCount += 1
                        foundBalls.append([x, y, w, h])
                        cv2.putText(f_img, "Bola", (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.8,  (255,255,255), 1)
        cv2.putText(f_img, f"Ball count: {ballCount}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255,255,255), 2)
        if ballCount == 1:
            x, y, w, h = foundBalls[0][0], foundBalls[0][1], foundBalls[0][2], foundBalls[0][3]
            cv2.putText(imgCropped, "Bola", (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (0,0,255), 1)
            return foundBalls[0]
    coloredBalls = stackImages(0.4, [[filteredImgs[0], filteredImgs[1], filteredImgs[2], filteredImgs[3]], [filteredImgs[4], filteredImgs[5], filteredImgs[6], filteredImgs[7]]])
    #return coloredBalls

def lineEquation(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    m = (y2-y1)/(x2-x1)
    n = y1-(m*x1)
    return m, n

def dottedLine(img,pt1,pt2,color):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,15):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)
    for p in pts:
            cv2.circle(img,p,3,color,-1)
            #cv2.putText(img, f"{p}", p, cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0))

def trajectoriaPrediction(taco, cueBall, coloredBalls):
    try:
        #cv2.line(imgCropped, (taco[0]+taco[2]//2, taco[1]+taco[3]//2), (cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2), (255,0,0), 2)
        cv2.circle(imgCropped, (cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2), cueBall[0]+cueBall[2]//2, (200,0,200), 1)
        #Cue ball to colored ball
        m1, n1 = lineEquation([taco[0]+taco[2]//2, taco[1]+taco[3]//2], [cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2])
        x1 = (coloredBalls[0]+coloredBalls[2]//2)+1
        y1 = int((m1*x1) + n1)
        dottedLine(imgCropped, (cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2), (x1, y1), (200,200,200))
        cv2.circle(imgCropped, (x1, y1), 5, (200,200,200), cv2.FILLED)

        #Colored ball to hole
        m2, n2 = lineEquation([x1, y1], [coloredBalls[0]+coloredBalls[2]//2, coloredBalls[1]+coloredBalls[3]//2])
        if cueBall[0] < coloredBalls[0]:
            x2 = 505
        else:
            x2 = 26
        y2 = int((m2*x2) + n2)
        if y2 > 346:
            y2 = 346
            x2 = int((346-n2)/m2)
        elif y2 < 51:
            y2 = 51
            x2 = int((51-n2)/m2)
        dottedLine(imgCropped, (x1, y1), (x2, y2), (0,0,150))

    except TypeError:
        pass


frameWidth = 960
frameHeight = 540
cap = cv2.VideoCapture("resources/shots.mp4")
while True:
    success, frame = cap.read()
    imgRaw = cv2.resize(frame, (frameWidth, frameHeight))
    imgCropped = imgRaw[10:460,80:881]
    cv2.imwrite("imgColors.png", imgCropped)
    taco = findTaco(imgCropped)
    cueBall = findCueBall(imgCropped)
    coloredBalls = findColoredBalls(imgCropped)
    #filterImg = findColoredBalls(imgCropped)
    #trajectoriaPrediction(taco, cueBall, coloredBalls)
    #finalImg = stackImages(0.7, [imgCropped, taco])
    cv2.imshow("Result", imgCropped)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break