import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)
vid.set(10,200)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevT = 0
currT = 0




while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks :
        for handLms in results.multi_hand_landmarks :
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id==12:
                    cv2.circle(img,(cx,cy),18,(255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)

    currT = time.time()
    fps = 1/(currT-prevT)
    prevT = currT

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)



    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
