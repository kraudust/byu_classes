import cv2
import numpy as np
from copy import deepcopy
from pdb import set_trace as pause

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

    def calc_optical_flow(self):
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
            self.frame_old = deepcopy(self.frame_new)
            self.p0 = np.reshape(p_correct[0:2,:],(-1,1,2))


class mean_cam_kalman():
    def __init__(self,video_path):
        # Initialize Camera
        self.cap = cv2.VideoCapture(video_path)

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")

        ret,frame = self.cap.read()

        # Get ROI
        self.track_window = cv2.selectROI("Image", frame, False, False)
        cx = self.track_window[0]
        cy = self.track_window[1]
        w = self.track_window[2]
        h = self.track_window[3]
        while(self.cap.isOpened()):
            ret,frame = self.cap.read()
            if ret == True:
                cv2.imshow('roi', frame[cy:cy+h, cx:cx+w, :])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # def display_image(self):
    #     while(self.cap.isOpened()):
    #         ret, frame = self.cap.read()
    #         if ret == True:
    #             cv2.imshow('frame', frame)
    #         # Press Q on keyboard to  exit
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #           break
        # # Convert to HSV
        # hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

        # # Get Hue mask and Histogram
        # mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.)),np.array((180.,255.,255.)))
        # self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        # cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

        # # Criteria
        # self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # # Setup Camshift stuff
        # self.setup_cams()
        
        # # Kalman Stuff
        # self.dt = 1.0 / 30.0
        # self.kalman = cv2.KalmanFilter(4,2,0)
        # self.kalman.transitionMatrix = np.array([[1., 0., self.dt, 0.],
        #                                          [0., 1., 0., self.dt],
        #                                          [0., 0., 1., 0.],
        #                                          [0., 0., 0., 1.]])
        # self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.],
        #                                           [0., 1., 0., 0.]])
        # self.kalman.processNoiseCov = 1e2 * np.array([[self.dt**3/3., 0., self.dt**2/2., 0.],
        #                                               [0., self.dt**3/3., 0., self.dt**2/2.],
        #                                               [self.dt**2/2., 0., self.dt, 0.],
        #                                               [0., self.dt**2/2., 0., self.dt]])
        # self.kalman.measurementNoiseCov = 1e-5 * np.eye(2)
        # self.kalman.statePost = np.array([[self.track_window[0]],
        #                                   [self.track_window[1]],
        #                                   [0.],
        #                                   [0.]])
        # self.kalman.errorCovPost = 0.1 * np.eye(4)
        # self.meas = np.array([[0.],[0.]])



if __name__ == "__main__":

    video_location = '/home/kraudust/git/personal_git/byu_classes/robotic_vision/hw5_object_tracking/mv2_001.avi'
    # video_location = 0

    # KLT Kalman Filter
    kt = klt_kalman(video_location)
    kt.calc_optical_flow()
    # kt = mean_cam_kalman(video_location)
    kt.cap.release()
    cv2.destroyAllWindows()

