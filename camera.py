# camera.py

import math
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
        self.hat = cv2.imread("hats/01.png")
        self.visual = True

    # Destructor
    def __del__(self):
        self.video.release()

    # Function for creating landmark coordinate list
    def land2coords(self, landmarks, dtype="int"):
        coords = np.zeros((68,2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    # Function for overlaying the hat
    def transparent_overlay(self, src, overlay, pos=(0, 0), scale=1):
        """
        :param src:  input color background image
        :param overlay: transparent image (BGRA)
        :param pos: position where the image to be built
        :param scale: scale factor of transparent image
        :return: resultant image
        """
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape       # size of foreground
        rows, cols, _ = src.shape     # size of background
        y, x = pos[0], pos[1]       # position of overlay image

        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x+i >= rows or y+j >= cols:
                    continue
                alpha = float(overlay[i][j][3]/255.0)   # read the alpha channel
                src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
        return src

    # Function for rotating the image
    def rotate_about_center(self, src, angle, scale=1.):
        """
        :param src: input color image
        :param angle: input angle in degree
        :param scale: scale factor for rotation image
        :return:
        """
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians (x*pi/180)

        # calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w)) * scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w)) * scale

        # get the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)

        # calculate the move from old center to new center
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))

        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh)))), int(nw), int(nh)

    # Main function
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
            image = self.get_pose(image, landmarks)
            image = self.add_hat(image, landmarks)

        ret, jpg = cv2.imencode('.jpg', image)
        return jpg.tobytes()

    # Function for getting head pose
    def get_pose(self, frame, landmarks):
        # 27th, 28th, 29th, 30th
        (x1, y1) = landmarks[27]
        (x2, y2) = landmarks[28]
        (x3, y3) = landmarks[29]
        (x4, y4) = landmarks[30]

        if self.visual:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 0), 2)
            cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)

        return frame

    # Function for adding christmas hat
    # LONG WAT TO GO
    def add_hat(self, frame, landmarks):
        # 19th and 24th points
        (a1, b1) = landmarks[19]
        (a2, b2) = landmarks[24]
        (a3, b3) = landmarks[38]
        (a4, b4) = landmarks[43]

        if self.visual:
            cv2.line(frame, (a1, 2*b1-b3), (a2, 2*b2-b4), (255, 0, 0), 2)
            # the blue line is where a hat will be placed

        length = math.sqrt(pow(a2-a1, 2)+pow(2*b2-b4-2*b1+b3, 2))
        w, h = self.hat.shape[:2]
        hat_new = cv2.resize(self.hat, (int(length), int(length*h/w)),interpolation=cv2.INTER_CUBIC)
        w_new, h_new = hat_new.shape[:2]
        #print(w, h, w_new, h_new)

        return frame
