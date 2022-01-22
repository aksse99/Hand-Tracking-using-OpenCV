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




while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks :
        for handLms in results.multi_hand_landmarks :
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)


    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
