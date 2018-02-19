# camera.py

import numpy as np
import cv2
import dlib


class VideoCamera(object):

    # Constructor
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        # Using OpenCV to open web-camera
        # Notice the index of camera

        # loading dlib's Hog Based face detector
        self.face_detector = dlib.get_frontal_face_detector()
        # loading dlib's 68 points-shape-predictor
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Destructor
    def __del__(self):
        self.video.release()

    # Function for creating landmark coordinate list
    def land2coords(self, landmarks, dtype="int"):
        coords = np.zeros((68,2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    # Function for getting frame and returning to flask
    def get_frame(self):
        success, image = self.video.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces
        face_boundaries = self.face_detector(image_gray, 0)

        for (enum, face) in enumerate(face_boundaries):
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            # predict and draw landmarks
            landmarks = self.landmark_predictor(image_gray, face)
            # convert co-ordinates to NumPy array
            landmarks = self.land2coords(landmarks)
            for (a, b) in landmarks:
                cv2.circle(image, (a, b), 2, (0, 255, 0), -1)
            image = self.add_hat(image, landmarks)

        ret, jpg = cv2.imencode('.jpg', image)
        return jpg.tobytes()

    # Function for adding christmas hat
    # LONG WAT TO GO
    def add_hat(self, frame, landmarks):
        # 19th and 24th points
        (a1, b1) = landmarks[19]
        (a2, b2) = landmarks[24]
        (a3, b3) = landmarks[38]
        (a4, b4) = landmarks[43]

        cv2.line(frame, (a1, 2*b1-b3), (a2, 2*b2-b4), (255, 0, 0), 3)
        # the blue line is where a hat will be placed

        return frame
