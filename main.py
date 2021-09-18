import cv2
import mediapipe as mp
import time


def init():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    previousTime = 0
    currentTime = 0
    mpHand = mp.solutions.hands
    hands = mpHand.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        # get the information of each framework only when a hand is detected.
        if results.multi_hand_landmarks:
            for handLandMarks in results.multi_hand_landmarks:
                for id, lm in enumerate(handLandMarks.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [0, 4, 8, 12, 16, 20]:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLandMarks, mpHand.HAND_CONNECTIONS)

        # compute the fps rate (frame per second)
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 250), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    init()


