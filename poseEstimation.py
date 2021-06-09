import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture('Sample/pexels.mp4')

p_time = 0
c_time = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = pose.process(img_RGB)

    if result.pose_landmarks:
        mDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)

            cv2.circle(img, (cx, cy), 10, (255,0,0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Adding FPS Text to the Video
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)