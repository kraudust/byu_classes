import cv2
import numpy as np
from pdb import set_trace as pause
from copy import deepcopy
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# initialize Kalman filter
Ts = 1.0/30.0
kalman = cv2.KalmanFilter(4,2)
kalman.transitionMatrix = np.array([[1., 0., Ts, 0.],[0., 1., 0., Ts], [0., 0., 1., 0.], [0., 0., 0., 1.]])
kalman.measurementMatrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
kalman.processNoiseCov = 1e2 * np.array

first_time_through = True
while(cap.isOpened()):
    if first_time_through == True:
        ret, frame = cap.read()
        gray_frame_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        showCrosshair = False
        fromCenter = False
        roi = cv2.selectROI("Image", frame, fromCenter, showCrosshair) #returns [top left corner x, top left corner y, width, height]
        roi_frame_gray = gray_frame_old[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]]
        p0 = cv2.goodFeaturesToTrack(roi_frame_gray, mask = None, **feature_params)
        p0[:,0,0] = p0[:,0,0] + roi[0]
        p0[:,0,1] = p0[:,0,1] + roi[1]
        first_time_through = False

    else:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frame_old, gray_frame, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # print(good_new)
        # pause()

        # Now update the previous frame and previous points
        gray_frame_old = deepcopy(gray_frame)
        p0 = good_new.reshape(-1,1,2)

        if ret == True:

            # write to file
            # out.write(frame)

            # Display the resulting frame
            # new_im = cv2.rectangle(frame,(min(good_new[:,0]), min(good_new[:,1])),(max(good_new[:,0]), max(good_new[:,1])), (0, 0, 255), 2)
            new_im = cv2.rectangle(frame,(good_new[:,0] - roi[2]/2.0, good_new[:,1] - roi[3]/2.0),(good_new[:,0] + roi[2]/2.0, good_new[:,1] + roi[3]/2.0), (0, 0, 255), 2)
            cv2.imshow('Original',new_im)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break

        # Break the loop
        else: 
            break
 
# When everything done, release the video capture object
cap.release()
# out.release()
 
# Closes all the frames
cv2.destroyAllWindows()
