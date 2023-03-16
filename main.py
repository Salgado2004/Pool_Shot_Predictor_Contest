import cv2
import math
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

def getCosSin(deg):
    angulo = math.radians(deg)
    seno = math.sin(angulo)
    cosseno = math.cos(angulo)

    if cosseno == 6.123233995736766e-17:
        cosseno = 0

    if seno == 6.123233995736766e-17:
        seno = 0

    return seno, cosseno

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

# Function to calculate the position of the holes
def findHoles():
    holes = [
        [15, 40, 65, 85],
        [372, 35, 412, 75],
        [725, 39, 795, 78],
        [25, 369, 69, 408],
        [377, 376, 420, 413],
        [725, 368, 766, 406]
    ]
    return holes

# Function to find the position of the cue 
def findTaco(img):
    croppedImg = img[90:, :]
    lower = np.array([77,44,159])
    upper = np.array([100,89,213])
    kernel = np.ones((5,5),np.uint8)
    imgFiltered = colorFilter(croppedImg, lower, upper)
    imgProcessed = imgProcessing(imgFiltered)
    dial = cv2.dilate(imgProcessed, kernel, iterations=1)
    thres = cv2.erode(dial, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = -1
    taco = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 20 and area < 150:
            cv2.drawContours(imgFiltered, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(imgFiltered, f"{w}, {h}" ,(x+w//2, y+h//2-10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(imgFiltered, f"{area}" ,(x+w//2, y+h//2+15),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 1)
            if objCor > 2:
                if w < 18 and w > 5 and h < 18 and h > 5:
                    if area > maxArea:
                        maxArea = area
                        taco = [x, y+90, w, h]
    if taco:
        cv2.rectangle(imgFiltered, (taco[0], taco[1]-90), (taco[0]+taco[2], taco[1]-90+taco[3]), (255,255,255), 2)
        cv2.putText(imgCropped, "Taco", (taco[0]+(taco[2]//2)-10, taco[1]+(taco[3]//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (0,0,0), 1)
        return taco
    #return imgFiltered

# Function to find the position of the cue ball
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
                return [x+38, y+56, w, h]

# Function to find the position of the colored balls
def findColoredBalls(img):
    croppedImg = img[110:375, 38:770]
    kernel = np.ones((5,5),np.uint8)
    lowerValues = [[13,133,132], [74, 33, 71], [61, 36, 70], [82, 71, 72], [0, 98, 70], [61, 36, 70], [120,53,116], [0, 0, 0]]
    upperValues = [[66,255,255], [123, 255, 255], [79, 232, 255], [125, 255, 255], [17, 255, 255], [79, 232, 255], [179,255,255], [179, 255, 255]]
    filteredImgs = []
    for x in range(8):
        lower = np.array(lowerValues[x])
        upper = np.array(upperValues[x])
        filteredImgs.append(colorFilter(croppedImg, lower, upper))
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
                    if h > 15 and h < 38 and w > 15 and w < 38 and (w-h) > -7 and (w-h) < 7:
                        ballCount += 1
                        foundBalls.append([x, y, w, h])
                        if w//2 > 15:
                            w = 15
                        cv2.circle(f_img, (x+w//2,y+h//2), (w//2), (255,255,0), 2)
                        cv2.putText(f_img, "Bola", (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.8,  (255,255,255), 1)
        cv2.putText(f_img, f"Ball count: {ballCount}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255,255,255), 2)
        if ballCount == 1:
            x, y, w, h = foundBalls[0][0], foundBalls[0][1], foundBalls[0][2], foundBalls[0][3]
            cv2.putText(imgCropped, "Bola", (x+(w//2)+28, y+(h//2)+88), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (0,0,255), 1)
            return [x+38, y+110, w, h]
    coloredBalls = stackImages(0.46, [[filteredImgs[0], filteredImgs[1], filteredImgs[2], filteredImgs[3]], [filteredImgs[4], filteredImgs[5], filteredImgs[6], filteredImgs[7]]])
    #return coloredBalls

# Function to detect the point of the cue that hits the ball
def getHitPoint(taco, cueBall, averageRadius, hitPoints):
    tacoPoints = []
    hitPoint = []
    cueBallX = cueBall[0]+cueBall[2]//2
    cueBallY = cueBall[1]+cueBall[3]//2

    averageRadius.append((taco[2]//2+taco[3]//2)//2)
    radius = 0
    for r in averageRadius:
        radius += r
    radius = radius//(len(averageRadius))

    oX = taco[0]+taco[2]//2
    oY = taco[1]+taco[3]//2
    for ang in range(0, 360):
        seno, cosseno = getCosSin(ang)
        pX = int(cosseno*radius)
        pY = int(seno*radius)
        tacoPoints.append([oX+pX, oY+pY])

    minDistance = 1000000
    for point in tacoPoints:
        distance = math.sqrt(math.pow(cueBallX-point[0], 2) + math.pow(cueBallY-point[1], 2))
        if distance < minDistance:
            minDistance = distance
            hitPoint = point

    hitPoints.append(hitPoint)
    sumX = 0
    sumY = 0
    for point in hitPoints:
        sumX += point[0]
        sumY += point[1]
    hitPoint = [sumX//len(hitPoints), sumY//len(hitPoints)]
    return hitPoint

# Function to calculate de line between two points
def lineEquation(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    try:
        m = (y2-y1)/(x2-x1)
    except ZeroDivisionError:
        m = (y2-y1)/(x2+1-x1)
    n = y1-(m*x1)
    return m, n

# Function to draw a dotted line
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

# Detect the collision point between the cue ball and the colored ball
def detectCollision(cueBall, coloredBall):   
    cueBallList = []
    coloredBallList = []

    radius = (cueBall[2]-cueBall[0])//2
    oX = cueBall[0]+(cueBall[2]-cueBall[0])//2
    oY = cueBall[1]+(cueBall[3]-cueBall[1])//2
    for ang in range(0, 360):
        seno, cosseno = getCosSin(ang)
        pX = int(cosseno*radius)
        pY = int(seno*radius)
        cueBallList.append([oX+pX, oY+pY])

    radius = (coloredBall[2]-coloredBall[0])//2
    oX = coloredBall[0]+(coloredBall[2]-coloredBall[0])//2
    oY = coloredBall[1]+(coloredBall[3]-coloredBall[1])//2
    for ang in range(0, 360):
        seno, cosseno = getCosSin(ang)
        pX = int(cosseno*radius)
        pY = int(seno*radius)
        coloredBallList.append([oX+pX, oY+pY])

    collisionPoints = []
    for point in cueBallList:
        if point in coloredBallList:
            collisionPoints.append(point)

    if len(collisionPoints) > 0:
        xPt = 0
        yPt = 0
        for point in collisionPoints:
            xPt += point[0]
            yPt += point[1]
        collisionPt = [xPt//len(collisionPoints), yPt//len(collisionPoints)]
        cv2.circle(imgCropped, (collisionPt[0], collisionPt[1]), 8, (0,200,200), cv2.FILLED)
        return True, collisionPt
    return False, []

def bouncePrediction(point, radius, holes):
    color = (0,0,200)
    inHole = False

    for hole in holes:
        if point[0] - radius >= hole[0] and point[1] - radius >= hole[1] and point[0] + radius <= hole[2] and point[1] + radius <= hole[3]:
            inHole = True
            color = (0,200,0)

    return color, inHole

def pathPrediction(collisionPoint, coloredBall, paths, holes):
    # Colored ball path
    ballCenter = [coloredBall[0]+coloredBall[2]//2, coloredBall[1]+coloredBall[3]//2]
    m2, n2 = lineEquation(collisionPoint, [ballCenter[0]+1, ballCenter[1]+1])
    
    if collisionPoint[0] > coloredBall[0]+coloredBall[2]//2:
        last_x = 30
    else:
        last_x = 790

    for i in range(0,2):
        x2 = last_x
        y2 = int((m2*x2)+n2)

        if y2 >= 390:
            y2 = 390
            x2 = int((y2-n2)/m2)
        if y2 <= 60:
            y2 = 60
            x2 = int((y2-n2)/m2)
        if y2 > 75 and y2 < 350 and x2 >= 765:
            x2 = 765
            y2 = int((m2*x2)+n2)
            last_x = 30
        if y2 > 75 and y2 < 350 and x2 <= 35:
            x2 = 35
            y2 = int((m2*x2)+n2)
            last_x = 765
        paths.append([x2, y2])
        color, inHole = bouncePrediction(paths[-1], 12, holes)
        if inHole:
            return paths, color, inHole
        else:
            m2 = -m2
            n2 = y2-(m2*x2)

    return paths, color, inHole

# Control all the calculations used for the prediction
def shotPrediction(hitPoint, cueBall, coloredBalls, holes):
    try:
        #Cue ball to colored ball
        m1, n1 = lineEquation([hitPoint[0], hitPoint[1]], [cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2])
        
        points = []
        x_last = (coloredBalls[0]+coloredBalls[2]//2)
        x1, y1 = x_last, int((m1*x_last) + n1)
        if x_last >= cueBall[0]+cueBall[2]//2:
            step = 1
        else:
            step = -1
        for x in range(cueBall[0]+cueBall[2]//2, x_last, step):
            y = int((m1*x) + n1)
            points.append([x, y])

        for point in points:
            bbox = [point[0]-cueBall[2]//2, point[1]-cueBall[3]//2, point[0]+cueBall[2]//2, point[1]+cueBall[3]//2,]
            collision, collisionPoint = detectCollision(bbox, [coloredBalls[0],coloredBalls[1], coloredBalls[0]+coloredBalls[2], coloredBalls[1]+coloredBalls[3]])
            if collision:
                x1, y1 = collisionPoint[0], collisionPoint[1]
                paths = [[coloredBalls[0]+coloredBalls[2]//2, coloredBalls[1]+coloredBalls[3]//2]]
                paths, color, inHole = pathPrediction(collisionPoint, coloredBalls, paths, holes)
                for i, path in enumerate(paths):
                    if i == 0:
                        pass
                    else:
                        dottedLine(imgCropped, (paths[i-1][0], paths[i-1][1]), (path[0], path[1]),  color)
                        cv2.circle(imgCropped, (path[0], path[1]), 10, color, cv2.FILLED)
                    if inHole:
                        cv2.rectangle(imgCropped, (80, 395), (280,440), color, cv2.FILLED)
                        cv2.putText(imgCropped, "Prediction: In", (85, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
                    else:
                        cv2.rectangle(imgCropped, (80, 395), (280,440), color, cv2.FILLED)
                        cv2.putText(imgCropped, "Prediction: Out", (85, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
                
                print("Bola branca: ", cueBall)
                print("Taco: ", hitPoint)
                print("Bola colorida: ", coloredBalls)
                print("Ponto de colisão: ", collisionPoint)
                print("Resultado: ", inHole)
                print("\n")
                break

        dottedLine(imgCropped, (cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2), (x1, y1), (200,200,200))
        cv2.circle(imgCropped, (x1, y1), 5, (200,200,200), cv2.FILLED)

    except TypeError:
        pass

cap = cv2.VideoCapture("resources/shots.mp4")
frameWidth = 960
frameHeight = 540
size = (frameWidth, frameHeight)

result = cv2.VideoWriter('resources/shotsProcessed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

holes = findHoles()
hitPoints = []
averageRadius = []
lastSpot = []
prediction = True

shotIndex = 1
while True:
    success, frame = cap.read()
    imgRaw = cv2.resize(frame, (frameWidth, frameHeight))
    imgCropped = imgRaw[10:460,80:881]

    # Detect the objects 
    taco = findTaco(imgCropped)
    cueBall = findCueBall(imgCropped)
    coloredBalls = findColoredBalls(imgCropped)
    #filterImg = findColoredBalls(imgCropped)

    # Start the calculations
    if taco and cueBall and coloredBalls:
        if not(lastSpot):
            lastSpot.append([cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2])
            lastSpot.append([cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2])
        else:
            lastSpot.append([cueBall[0]+cueBall[2]//2, cueBall[1]+cueBall[3]//2])

        print("\nLast spots:", lastSpot[-2], lastSpot[-1])
        difference = lambda a, b : math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))
        if difference(lastSpot[-1], lastSpot[-2]) >= 2:
            prediction = False
            cv2.putText(imgRaw, f"Bola em movimento - Tacada {shotIndex}/10", (10,25), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,0), 2)
        elif len(lastSpot) > 2:
            if difference(lastSpot[-2], lastSpot[-3]) >= 2 and difference(lastSpot[-1], lastSpot[-2]) < 2:
                prediction = True
                hitPoints = []
                shotIndex += 1
        
        
        if prediction:
            hitPoint = getHitPoint(taco, cueBall, averageRadius, hitPoints)
            shotPrediction(hitPoint, cueBall, coloredBalls, holes)
    #finalImg = stackImages(0.8, [imgCropped, taco])
    cv2.imshow("Result", imgRaw)
    result.write(imgRaw)
    if cv2.waitKey(75) & 0xFF == ord('q'):
        break

result.release()