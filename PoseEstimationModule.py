import cv2
import time
import mediapipe as mp


class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

        self.mDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):

        self.img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.result = self.pose.process(self.img_RGB)

        if self.result.pose_landmarks:
            if draw:
                self.mDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPositions(self, img, draw=True):
        lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('Sample/pexels.mp4')

    detector = PoseDetector()

    p_time = 0
    c_time = 0

    while True:
        success, img = cap.read()

        img = detector.findPose(img)

        lmList = detector.findPositions(img)

        print(lmList)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Adding FPS Text to the Video
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
