import cv2
import numpy as np
from copy import deepcopy
from pdb import set_trace as pause
from scipy.stats import mode

class mean_cam_kalman():
    def __init__(self, tracker_type, frame):
        self.tracker_type = tracker_type
        # Initialize Camera
        # self.cap = cv2.VideoCapture(video_path)

        # # Check if camera opened successfully
        # if (self.cap.isOpened()== False):
        #     print("Error opening video stream or file")

        # ret,frame = self.cap.read()

        # Get window to track
        # self.track_window = cv2.selectROI("Image", frame, False, False)
        # # print(self.track_window)
        # cx, cy, w, h = self.track_window
        cx = 0
        cy = 0
        w = 640
        h = 480
        self.track_window = (cx, cy, w, h)
        # cx = self.track_window[0]
        # cy = self.track_window[1]
        # w = self.track_window[2]
        # h = self.track_window[3]

        # # Set up region of interest for tracking
        roi = frame[cy:cy+h, cx:cx+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        most = np.float32(mode(hsv_roi[:,:,0], axis=None))[0][0]
        print most
        # mask = cv2.inRange(hsv_roi, np.array((most-20., 50.,0.)), np.array((most+20.,255.,255.)))
        # mask = cv2.inRange(hsv_roi, np.array((0,0,0.)), np.array((180,255.,255.)))
        # res = cv2.bitwise_and(roi, roi, mask=mask)
        # cv2.imshow('mask',mask)
        # cv2.imshow('mask applied',res)
        # cv2.waitKey(30)
        # self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        # cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)  # should it be 255?

        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 5)

        # initialize Kalman Filter
        self.Ts = 1./30. # timestep
        self.kalman = cv2.KalmanFilter(4,2) # kalman filter object
        self.kalman.transitionMatrix = np.array([   [1., 0., self.Ts, 0.],
                                                    [0., 1., 0., self.Ts],
                                                    [0., 0., 1., 0.],
                                                    [0., 0., 0., 1.]], np.float32) # state transition matrix (discrete)
        self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], np.float32)
        self.kalman.processNoiseCov = 1e1 * np.array([[self.Ts**3/3., 0., self.Ts**2/2., 0.],
                                                      [0., self.Ts**3/3., 0., self.Ts**2/2.],
                                                      [self.Ts**2/2., 0., self.Ts, 0.],
                                                      [0., self.Ts**2/2., 0., self.Ts]], np.float32)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float32)
        self.kalman.statePost = np.array([[cx],
                                          [cy],
                                          [0.],
                                          [0.]], dtype=np.float32)
        self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)

    def run_tracking(self, frame):
        # while(True):
            # ret, frame = self.cap.read()
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift or camshift to get the new location
        if self.tracker_type == 'mean':
            ret, self.track_window = cv2.meanShift(frame, self.track_window, self.term_crit)
        elif self.tracker_type == 'cam':
            ret, self.track_window = cv2.CamShift(frame, self.track_window, self.term_crit)

        # Update with Kalman Filter
        p_predict = self.kalman.predict()
        p_correct = self.kalman.correct(np.array([[self.track_window[0]],[self.track_window[1]]], np.float32))

        # # Draw track window and display it
        # cx, cy, w, h = self.track_window
        # new_im = cv2.rectangle(frame, (cx,cy), (cx+w, cy+h), (0,0,255), thickness=2)
        # cv2.imshow('Image', new_im)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     # break
        #     pass
