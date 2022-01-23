import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks :
            for handLms in results.multi_hand_landmarks :
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img





def main():
    prevT = 0
    currT = 0
    vid = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = vid.read()
        img = detector.findHands(img)

        currT = time.time()
        fps = 1/(currT-prevT)
        prevT = currT

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break

if __name__ == '__main__':
    main()