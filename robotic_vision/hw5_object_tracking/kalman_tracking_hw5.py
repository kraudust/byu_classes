import cv2
import numpy as np
from copy import deepcopy
from pdb import set_trace as pause
from scipy.stats import mode

class klt_kalman():
    def __init__(self, video_path):

        # Open Camera or Video
        self.cap = cv2.VideoCapture(video_path)

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 1,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

        # params for lucase kanade optical flow
        self.lk_params = dict(  winSize  = (15, 15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.p1 = None # p0 passed through optical flow

        ret, self.frame_old = self.cap.read()
        self.cx, self.cy, self.w, self.h = cv2.selectROI("Image", self.frame_old, False, False)
        self.cx, self.cy, self.w, self.h = (255, 92, 26, 18)
        # print(self.cx, self.cy, self.w, self.h)
        self.frame_old = cv2.cvtColor(self.frame_old, cv2.COLOR_BGR2GRAY)

        # self.roi = self.frame_old[self.cx:int(self.cx + self.w/2), self.cy:int(self.cy + self.h/2.)]
        self.roi = self.frame_old[int(self.cy):int(self.cy + self.h/2), int(self.cx):int(self.cx + self.w/2.)]
        self.p0 = cv2.goodFeaturesToTrack(self.roi, mask = None, **self.feature_params)
        self.p0[:,0,0] = self.p0[:,0,0] + self.cx
        self.p0[:,0,1] = self.p0[:,0,1] + self.cy
        self.p0 = np.float32(self.p0)

        # initialize Kalman Filter
        self.Ts = 1./30. # timestep
        self.kalman = cv2.KalmanFilter(4,2) # kalman filter object
        self.kalman.transitionMatrix = np.array([   [1., 0., self.Ts, 0.],
                                                    [0., 1., 0., self.Ts], 
                                                    [0., 0., 1., 0.], 
                                                    [0., 0., 0., 1.]], np.float32) # state transition matrix (discrete)
        self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], np.float32)
        self.kalman.processNoiseCov = 1e+2 * np.array([[self.Ts**3/3., 0., self.Ts**2/2., 0.],
                                                      [0., self.Ts**3/3., 0., self.Ts**2/2.],
                                                      [self.Ts**2/2., 0., self.Ts, 0.],
                                                      [0., self.Ts**2/2., 0., self.Ts]], np.float32)
        self.kalman.measurementNoiseCov = 1e-5 * np.eye(2, dtype=np.float32)
        self.kalman.statePost = np.array([[self.p0[0][0][0]],
                                          [self.p0[0][0][1]],
                                          [0.],
                                          [0.]], dtype=np.float32)
        self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)

    def run_tracking(self):
        while(self.cap.isOpened()):
            # Capture Frame
            ret, frame = self.cap.read()
            self.frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.frame_old, self.frame_new, self.p0, None, **self.lk_params)

            # Update with Kalman Filter
            p_predict = self.kalman.predict()
            # self.meas[0,0] = good_new[0,0]
            # self.meas[1,0] = good_new[0,1]
            p_correct = self.kalman.correct(np.array([[self.p1[0,0,0]], [self.p1[0,0,1]]], np.float32))

            if ret == True:
                pt1 = (int(p_correct[0]-self.w/2), int(p_correct[1]-self.h/2))
                pt2 = (int(p_correct[0]+self.w/2), int(p_correct[1]+self.h/2))
                new_im = cv2.rectangle(frame, pt1, pt2, (0,0,255), thickness=2)
                cv2.imshow('Original',new_im)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break

            # Break the loop
            else: 
                break

            # Save the current image and point as the previous image and point
            self.frame_old = deepcopy(self.frame_new)
            self.p0 = np.reshape(p_correct[0:2,:],(-1,1,2))

class mean_cam_kalman():
    def __init__(self,video_path, tracker_type):
        self.tracker_type = tracker_type
        # Initialize Camera
        self.cap = cv2.VideoCapture(video_path)

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        ret,frame = self.cap.read()

        # Get window to track
        self.track_window = cv2.selectROI("Image", frame, False, False)
        # self.track_window = (257, 96, 18, 11)
        # print(self.track_window)
        cx, cy, w, h = self.track_window
        # cx = self.track_window[0]
        # cy = self.track_window[1]
        # w = self.track_window[2]
        # h = self.track_window[3]

        # Set up region of interest for tracking
        roi = frame[cy:cy+h, cx:cx+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        most = np.float32(mode(hsv_roi[:,:,0], axis=None))[0][0]
        mask = cv2.inRange(hsv_roi, np.array((most-20., 50.,0.)), np.array((most+20.,255.,255.)))
        # mask = cv2.inRange(hsv_roi, np.array((0,0,0.)), np.array((180,255.,255.)))
        # res = cv2.bitwise_and(roi, roi, mask=mask)
        # cv2.imshow('mask',mask)
        # cv2.imshow('mask applied',res)
        # cv2.waitKey(30)
        # pause()
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)  # should it be 255?

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

    def run_tracking(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

                # apply meanshift or camshift to get the new location
                if self.tracker_type == 'mean':
                    ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
                elif self.tracker_type == 'cam':
                    ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

                # Update with Kalman Filter
                p_predict = self.kalman.predict()
                p_correct = self.kalman.correct(np.array([[self.track_window[0]],[self.track_window[1]]], np.float32))

                # Draw track window and display it
                cx, cy, w, h = self.track_window
                new_im = cv2.rectangle(frame, (cx,cy), (cx+w, cy+h), (0,0,255), thickness=2)
                cv2.imshow('Image', new_im)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break
            else:
                break

class back_sub_kalman():
    def __init__(self, video_path):
        # Initialize Camera
        self.cap = cv2.VideoCapture(video_path)

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        ret,self.frame_old = self.cap.read()
        self.blur = 5
        self.frame_old = cv2.cvtColor(self.frame_old, cv2.COLOR_BGR2GRAY)
        # self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.kernel = np.ones((2,2), np.uint8)
        # self.frame_old = cv2.GaussianBlur(self.frame_prev, (self.blur, self.blur), 0)

        # # initialize Kalman Filter
        # self.Ts = 1./30. # timestep
        # self.kalman = cv2.KalmanFilter(4,2) # kalman filter object
        # self.kalman.transitionMatrix = np.array([   [1., 0., self.Ts, 0.],
        #                                             [0., 1., 0., self.Ts], 
        #                                             [0., 0., 1., 0.], 
        #                                             [0., 0., 0., 1.]], np.float32) # state transition matrix (discrete)
        # self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], np.float32)
        # self.kalman.processNoiseCov = 1e1 * np.array([[self.Ts**3/3., 0., self.Ts**2/2., 0.],
        #                                               [0., self.Ts**3/3., 0., self.Ts**2/2.],
        #                                               [self.Ts**2/2., 0., self.Ts, 0.],
        #                                               [0., self.Ts**2/2., 0., self.Ts]], np.float32)
        # self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float32)
        # self.kalman.statePost = np.array([[cx],
        #                                   [cy],
        #                                   [0.],
        #                                   [0.]], dtype=np.float32)
        # self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
    def run_tracking(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                self.frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(self.frame_new, self.frame_old)
                thresh_diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
                new_im = cv2.erode(thresh_diff, None, iterations = 1)
                new_im = cv2.dilate(new_im, self.kernel, iterations = 3)
                cv2.imshow('diff', new_im)
                cv2.imshow('orig', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break

            # Break the loop
            else: 
                break
            self.frame_old = deepcopy(self.frame_new)

if __name__ == "__main__":

    video_location = '/home/kraudust/git/personal_git/byu_classes/robotic_vision/hw5_object_tracking/mv2_001.avi'
    # video_location = 0

    # KLT Kalman Filter
    # kt = klt_kalman(video_location)
    # kt = mean_cam_kalman(video_location, 'cam')
    kt = back_sub_kalman(video_location)
    kt.run_tracking()
    kt.cap.release()
    cv2.destroyAllWindows()

