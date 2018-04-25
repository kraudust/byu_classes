#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from pdb import set_trace as pause
import argparse
import struct
import sys
import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from recursive_ls import rls
from copy import deepcopy, copy
import cv2
from threading import RLock
from mean_shift import mean_cam_kalman as track
import time
import yaml

class BaxDesPoseRight():
    def __init__(self):
        self.lock = RLock()
        rospy.init_node('bax_des_pos', anonymous=True)

        # Publishers
        self.pub_des_pos = rospy.Publisher('/bax_des_right_pose', PoseStamped, queue_size = 1)

        # Variables
        self.des_pos = np.array([0.,0.,0.]) # cartesian position to catch ball at
        self.ball_pos = np.array([0., 0., 0.]) # the cartesian position of the ball relative to right camera frame
        self.ball_pos_b = np.matrix([[0.],[0.],[0.],[1.]])
        self.ball_pos_imL = np.zeros((1,1,2), dtype=np.float32)
        self.ball_pos_imR = np.zeros((1,1,2), dtype=np.float32)
        self.xp = 0.76 # plane I want to catch the ball on
        th = 10.*np.pi/180.#angle of the computer relative to vertical
        Rb_i = np.matrix([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        Ri_c = np.matrix([[1., 0., 0.], [0., np.cos(th-np.pi/2.), -np.sin(th-np.pi/2.)], [0., np.sin(th-np.pi/2.), np.cos(th-np.pi/2.)]])
        Rb_c = Rb_i*Ri_c
        self.Tb_c = np.matrix(np.zeros((4,4)))
        self.Tb_c[0:3, 0:3] = Rb_c
        self.Tb_c[0,3] = 0.5
        self.Tb_c[3,3] = 1.

        # Open Cameras, and start tracker
        self.Lcap = cv2.VideoCapture(1)
        self.Rcap = cv2.VideoCapture(0)
        for i in xrange(30):
            retL, self.left_cam_img = self.Lcap.read()
        # self.trackerL = track('mean', self.left_cam_img)
        self.trackerL = track('cam', self.left_cam_img)
        for i in xrange(30):
            retR, self.right_cam_img = self.Rcap.read()
        # self.trackerR = track('mean', self.right_cam_img)
        self.trackerR = track('cam', self.right_cam_img)

        # Load Camera Calibrations
        laptop_stream = open('black_laptop_calibration_data.yaml', 'r')
        laptop_dict = yaml.load(laptop_stream)
        self.Kl = np.reshape(np.array(laptop_dict['camera_matrix']['data'], dtype = np.float32), (3,3)) # Laptop camera calibration matrix
        self.dl = np.array(laptop_dict['distortion_coefficients']['data'], dtype = np.float32) # Laptop distortion parameters

        webcam_stream = open('phil_camera_calibration_data.yaml', 'r')
        webcam_dict = yaml.load(webcam_stream)
        self.Kw = np.reshape(np.array(webcam_dict['camera_matrix']['data'], dtype = np.float32), (3,3)) # Webcam camera calibration matrix
        self.dw = np.array(webcam_dict['distortion_coefficients']['data'], dtype = np.float32) # Webcam distortion parameters

        self.P1 = np.eye(4) # From the camera perspective, the transform to the right camera (in this case the right camera is the world frame)
        # self.P2 = np.array([[0.9369381471340504, -0.010378654462746245, 0.349341082577694,-0.6580634975313188], [0.0043768430092186485, 0.9998290293858212, 0.01796539013435,-0.13596629522983766],[-0.3494678120947406, -0.015303448269905184, 0.9368233839848178, -0.8915175344029096]])
        self.P2 = np.array([[0.9369381471340504, -0.010378654462746245, 0.349341082577694,-0.87], [0.0043768430092186485, 0.9998290293858212, 0.01796539013435,-0.13596629522983766],[-0.3494678120947406, -0.015303448269905184, 0.9368233839848178, -0.8915175344029096]])
        P2inv = np.zeros((3,4))
        P2inv[0:3,0:3] = self.P2[0:3,0:3].T
        P2inv[0:3,3] = np.dot(-self.P2[0:3,0:3].T, self.P2[0:3,3])
        self.P2 = deepcopy(P2inv)

        # Tools
        self.quadratic_fit = rls('quadratic', np.array((-0.01176, 0.1265, 0.4118)))
        self.linear_fit = rls('linear', np.array((0.05, -0.5)))


    def get_ball_position(self):
        retL, self.left_cam_img = self.Lcap.read()
        retR, self.right_cam_img = self.Rcap.read()

        # blur image to reduce noise
        blurL = cv2.GaussianBlur(self.left_cam_img, (5,5),0)
        blurR = cv2.GaussianBlur(self.right_cam_img, (5,5),0)

        # Convert BGR to HSV
        hsvL = cv2.cvtColor(blurL, cv2.COLOR_BGR2HSV)
        # hsvL = cv2.cvtColor(self.left_cam_img, cv2.COLOR_BGR2HSV)
        hsvR = cv2.cvtColor(blurR, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image for only blue colors
        # lower_blueL = np.array([106,50,50])
        # upper_blueL = np.array([115,255,255])
        # lower_blueR = np.array([106,50,50])
        # upper_blueR = np.array([115,255,255])
        lower_blueL = np.array([100,50,50])
        upper_blueL = np.array([120,255,255])
        lower_blueR = np.array([100,50,50])
        upper_blueR = np.array([120,255,255])
        # lower_green = np.array([3,100,100])
        # upper_green = np.array([10,255,255])

        # Threshold the HSV image to get only blue colors
        maskL = cv2.inRange(hsvL, lower_blueL, upper_blueL)
        maskR = cv2.inRange(hsvR, lower_blueR, upper_blueR)

        # Blur the mask
        bmaskL = cv2.GaussianBlur(maskL, (5,5),0)
        bmaskR = cv2.GaussianBlur(maskR, (5,5),0)

        # Erode and Dilate
        kernel = np.ones((5,5), np.uint8)
        num_edL = 2
        num_edR = 3
        erosionL = cv2.erode(bmaskL, kernel, iterations = num_edL)
        erosionR = cv2.erode(bmaskR, kernel, iterations = num_edR)
        dilationL = cv2.dilate(erosionL, kernel, iterations = num_edL)
        dilationR = cv2.dilate(erosionR, kernel, iterations = num_edR)

        cv2.imshow('ImageL', dilationL)
        cv2.imshow('ImageR', dilationR)

        # One method of tracking ----------------------------------------------------------------------
        # # Take the moments to get the centroid
        # momentsL = cv2.moments(dilationL)
        # momentsR = cv2.moments(dilationR)
        # m00L = momentsL['m00']
        # m00R = momentsR['m00']
        # centroid_xL, centroid_yL = None, None
        # centroid_xR, centroid_yR = None, None
        # if m00L != 0:
        #     centroid_xL = int(momentsL['m10']/m00L)
        #     centroid_yL = int(momentsL['m01']/m00L)
        # if m00R != 0:
        #     centroid_xR = int(momentsR['m10']/m00R)
        #     centroid_yR = int(momentsR['m01']/m00R)

        # # Assume no centroid
        # ctrL = (-1,-1)
        # ctrR = (-1,-1)

        # # Use centroid if it exists
        # if centroid_xL != None and centroid_yL != None:
        #     ctrL = (centroid_xL, centroid_yL)

        # if centroid_xR != None and centroid_yR != None:
        #     ctrR = (centroid_xR, centroid_yR)

        # # Put black circle in at centroid in image
        # cv2.circle(self.left_cam_img, ctrL, 10, (0,0,0), thickness=-1)
        # cv2.circle(self.right_cam_img, ctrR, 10, (0,0,0), thickness=-1)
        # cv2.imshow('TrackingL', self.left_cam_img)
        # cv2.imshow('TrackingR', self.right_cam_img)

        # Cam or meanshift ----------------------------------------------------------------------------
        # Display full-color image
        # cv2.imshow('GreenBallTracker', self.left_cam_img)
        self.trackerL.run_tracking(dilationL)
        self.trackerR.run_tracking(dilationR)
        cxL, cyL, wL, hL = self.trackerL.track_window
        cxR, cyR, wR, hR = self.trackerR.track_window
        new_imL = cv2.rectangle(self.left_cam_img, (cxL,cyL), (cxL+wL, cyL+hL), (0,0,255), thickness=2)
        new_imR = cv2.rectangle(self.right_cam_img, (cxR,cyR), (cxR+wR, cyR+hR), (0,0,255), thickness=2)
        cv2.imshow('TrackingL', new_imL)
        cv2.imshow('TrackingR', new_imR)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            pass
        ctrL = [cxL + wL/2., cyL + hL/2.]
        ctrR = [cxR + wR/2., cyR + hR/2.]

        # camera is 640x480
        self.ball_pos_imL[0,0,0] = ctrL[0]
        self.ball_pos_imL[0,0,1] = ctrL[1]
        self.ball_pos_imR[0,0,0] = ctrR[0]
        self.ball_pos_imR[0,0,1] = ctrR[1]

        pointR = cv2.undistortPoints(self.ball_pos_imR, self.Kl, self.dl)
        pointL = cv2.undistortPoints(self.ball_pos_imL, self.Kw, self.dw)
	# pointL = np.array([[ctrL[0]],[ctrL[1]]], np.float32)
        # pointL[0,0] = (pointL[0,0] - 320.)/320.
        # pointL[1,0] = (pointL[1,0] - 240.)/240.

	# pointR = np.array([[ctrR[0]],[ctrR[1]]], np.float32)
        # pointR[0,0] = (pointR[0,0] - 320.)/320.
        # pointR[1,0] = (pointR[1,0] - 240.)/240.
	# X = cv2.triangulatePoints(self.P1[:3], self.P2, pointL, pointR)
	X = cv2.triangulatePoints(self.P1[:3], self.P2, pointR, pointL)
        self.ball_pos[0] = X[0,0]/float(X[3])
        self.ball_pos[1] = X[1,0]/float(X[3])
        self.ball_pos[2] = X[2,0]/float(X[3])
        # self.ball_pos_b[0,0] = self.ball_pos[0]
        # self.ball_pos_b[1,0] = self.ball_pos[1]
        # self.ball_pos_b[2,0] = self.ball_pos[2]
        self.ball_pos_b = self.Tb_c*np.matrix([[self.ball_pos[0]],[self.ball_pos[1]],[self.ball_pos[2]],[1.]])

    def predict_ball_catch_position(self):
        # get predicted intersection of ball path with catching plane--------------------------------------
        self.quadratic_fit.update_variables(np.array([self.ball_pos_b[0,0], self.ball_pos_b[2,0]]))
        self.linear_fit.update_variables(np.array([self.ball_pos_b[0,0], self.ball_pos_b[1,0]]))
        a = self.quadratic_fit.est_var[0]
        bq = self.quadratic_fit.est_var[1]
        c = self.quadratic_fit.est_var[2]
        m = self.linear_fit.est_var[0]
        bl = self.linear_fit.est_var[1]
        self.des_pos[0] = self.xp
        self.des_pos[1] = m*self.xp + bl
        self.des_pos[2] = a*self.xp**2 + bq*self.xp + c

        # saturate the desired poses so he doesn't break himself
        if self.des_pos[1] > -0.2:
            self.des_pos[1] = -0.2
        elif self.des_pos[1] < -0.7:
            self.des_pos[1] = -0.7

        if self.des_pos[2] > 0.7:
            self.des_pos[2] = 0.7
        elif self.des_pos[2] < 0.2:
            self.des_pos[2] = 0.2

        print self.des_pos

        # self.des_pos = raw_input('Enter the desired position: ')
        # self.des_pos = np.fromstring(self.des_pos, dtype = float, count = -1, sep = ',')

    def publish_robot_ee_pose(self):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        des_pose_bax_right = {'right': PoseStamped(
            header=hdr,
            pose=Pose(
                # positions should be on plane [0.76, y, z] with y between 0 and -0.7 and z between 0.1 and 0.7
                position=Point(
                    x= self.des_pos[0],
                    y= self.des_pos[1],
                    z= self.des_pos[2],
                ),
                orientation=Quaternion(
                    x=-0.02826957,
                    y=0.72074924,
                    z=0.02521792,
                    w=0.69215997,
                ),
            ),
        )}
        self.pub_des_pos.publish(des_pose_bax_right['right'])

if __name__=='__main__':
    bax_right_pose = BaxDesPoseRight()
    i = 0
    while not rospy.is_shutdown():
        bax_right_pose.get_ball_position()
        bax_right_pose.predict_ball_catch_position()
        # if i == 20:
        bax_right_pose.publish_robot_ee_pose()
            # i = 0
        # else:
            # i = i + 1
