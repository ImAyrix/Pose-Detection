import cv2
import time
import PoseModule
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture("../PoseVideos/2.mp4")
pTime = 0
detector = PoseModule.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))
    img = detector.findPose(img)
    lmList = detector.getPosition(img)

    length = lmList[19][2]
    per = np.interp(int(length), [157, 285], [100, 0])
    poseBar = np.interp(length, [157, 285], [150, 400])

    cv2.putText(img, f"{int(per)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 103, 106), 3)
    cv2.rectangle(img, (50, int(poseBar)), (85, 400), (0, 182, 255), cv2.FILLED)
    cv2.rectangle(img, (50, 150), (85, 400), (240, 103, 106), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (425, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 103, 106), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
