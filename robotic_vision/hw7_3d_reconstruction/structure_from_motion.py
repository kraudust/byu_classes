import numpy as np
from Holodeck import Holodeck, Agents
# from Holodeck import Holodeck
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2
import scipy.io as sio
from copy import deepcopy
import matplotlib.pyplot as plt
from pdb import set_trace as pause
import pyqtgraph as pg

class visual_odom():
    def __init__(self):
        # positive axes
        # x - out the back
        # y - out the right wing
        # z - out the top
        # states
        self.init_plots = True
        self.R_cam_quad = np.array([[0.,0.,-1.],[1., 0., 0.], [0., -1., 0.]])
        self.get_first_states = True
        self.orientation = None
        self.phi = 0.0
        self.th = 0.0
        self.psi = 0.0
        self.velocity = None
        self.body_vel = None
        self.position = None
        self.state = {} # states dictionary containing velocity, orientation, position, and imu
        self.flag = 0
        self.env = Holodeck.make("UrbanCity")
        # self.env = Holodeck.make("EuropeanForest")
        self.command = np.array([0.0, 0.0, 0.0, 0.0]) # roll, pitch, yaw rate, altitude
        self.state, reward, terminal, _ = self.env.step(self.command)

        # Grab first image for optical flow
        self.image = self.state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

        # Convert to grayscale
        self.gray_old = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        self.gray_cur = None # current grascale image for optical flow

        self.detector = cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
        self.p0 = self.detector.detect(self.gray_old)
        self.p0 = np.array([x.pt for x in self.p0],dtype=np.float32)
        self.p1 = None

        # params for lucas kanade optical flow
        # self.lk_params = dict( winSize  = (15,15),
        #               maxLevel = 2,
        #               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_params = dict(winSize = (21,21),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def setup_pygame(self):
        pygame.init()
        size = [200, 200]
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("My Game")

    def get_states(self):
        # Velocity
        self.velocity = deepcopy(self.state[Sensors.VELOCITY_SENSOR]) # velocity in world frame
        self.velocity[0] = -self.velocity[0] # for some reason the position and velocities are in a left handed coordinate frame, this fixes it to line up with the orientation axes

        self.orientation = deepcopy(self.state[Sensors.ORIENTATION_SENSOR]) # orientation (a rotation matrix)
        self.orientation_est = deepcopy(self.orientation) # orientation (a rotation matrix)

        # Body Frame Velocities
        self.body_vel = np.matmul(self.orientation, np.divide(self.velocity, 100.0))

        # Positions
        self.position = deepcopy(self.state[Sensors.LOCATION_SENSOR]) # position in world frame
        self.position[0] = -self.position[0] # for some reason the position and velocities are in a left handed coordinate frame, this fixes it to line up with the orientation axes
        # imu = self.state[Sensors.IMU_SENSOR] # IMU
        self.image = deepcopy(self.state[Sensors.PRIMARY_PLAYER_CAMERA]) # camera
        self.gray_cur = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        # Find orientation states
        self.th = np.arcsin(-self.orientation[2,0]) # pitch
        self.phi = np.arcsin(self.orientation[2,1]/np.cos(self.th)) # roll
        self.psi = np.arcsin(self.orientation[1,0]/np.cos(self.th)) # yaw
        if self.get_first_states == True:
            self.position_est = deepcopy(self.position)
            self.get_first_states = False

    def calc_optical_flow(self):
        # Calculate new points that grids have moved to
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0, None, **self.lk_params)
        self.p1 = np.reshape(self.p1,(np.shape(self.p1)[0],1,2))
        self.p0 = np.reshape(self.p0,(np.shape(self.p0)[0],1,2))

        # # Select points where optical flow exists (i.e. inverse existed)
        self.good_new = self.p1[st==1]
        self.good_old = self.p0[st==1]

        # draw the tracks for visualization
        mask = np.zeros_like(self.image)
        for j,(new,old) in enumerate(zip(self.good_new,self.good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 1)
            image = cv2.circle(self.image,(a,b),2,[0,0,255],-1)
        img = cv2.add(image,mask)

        cv2.imshow('Optic Flow', img)
        self.gray_old = deepcopy(self.gray_cur)

        # Calculate new features to track in current image
        # pause()


    def compute_R_and_t(self):

        # E, mask = cv2.findEssentialMat(self.good_new, self.good_old, focal=1.0, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # E, mask = cv2.findEssentialMat(self.p0, self.p1, method=cv2.RANSAC, prob=0.999, threshold=0.1)
        E, mask = cv2.findEssentialMat(self.good_old, self.good_new, focal=256, pp=(256,256),method=cv2.RANSAC, prob=0.999, threshold=0.1)
        # R1, R2, t = cv2.decomposeEssentialMat(E)
        tmp, R1, t, tmp2 = cv2.recoverPose(E, self.good_old, self.good_new)
        if np.trace(R1) > 2.5:
            R = deepcopy(R1)
        # elif np.trace(R2) > 2.5:
        #     R = deepcopy(R2)
        else:
            R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # _, _, t, mask = cv2.recoverPose(E, self.p1, self.p0)
        # t[0] = t[0]


        #checks that x2*E*x1 = 0 and x2*that*R*x1 = 0
        # that = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        # x2 = np.append(self.good_new[0,:], 1).reshape(3,1)
        # x1 = np.append(self.good_old[0,:], 1).reshape(3,1)
        # print(np.matmul(x2.T, np.matmul(that, np.matmul(R, x1))))
        # print(np.matmul(x2.T, np.matmul(E, x1)), '\n')

        # Rpos = np.matmul(R, self.orientation)
        # self.position_est = self.position_est + np.matmul(t, Rpos)
        # diff =  np.matmul(self.orientation.T, np.matmul(self.R_cam_quad,t*np.linalg.norm(self.velocity)/10.))
        # diff[0] = -diff[0]


        # self.position_est = self.position_est + np.matmul(self.orientation, np.matmul(self.R_cam_quad,t*np.linalg.norm(self.velocity)/10.))
        # self.position_est = self.position_est + np.matmul(self.orientation_est, np.matmul(self.R_cam_quad,t*np.linalg.norm(self.velocity)/10.))
        # self.position_est = self.position_est + np.matmul(self.orientation, np.matmul(self.R_cam_quad,t*np.linalg.norm(self.velocity)/10.))
        # self.position_est = self.position_est + np.matmul(self.orientation,t*np.linalg.norm(self.velocity)/10.)
        # self.orientation_est = np.matmul(R, self.orientation_est)
        # self.position_est = self.position_est + diff 

        # print(R)
        # print(t)
        print(np.matmul(self.R_cam_quad, t))
        # print(np.linalg.norm(t))
        if len(self.good_new) < 300:
            self.p0 = self.detector.detect(self.gray_old)
            self.p0 = np.array([x.pt for x in self.p0],dtype=np.float32)
            # self.p0 = cv2.goodFeaturesToTrack(self.gray_old, mask = None, **self.feature_params)
        else:
            # self.p0 = np.reshape(deepcopy(self.good_new), (np.shape(self.good_new)[0], 1, np.shape(self.good_new)[1]))
            self.p0 = deepcopy(self.good_new)

    def send_commands(self,events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.command = self.command + np.array([0.0, -0.1, 0.0, 0.0])
                if event.key == pygame.K_DOWN:
                    self.command = self.command + np.array([0.0, 0.1, 0.0, 0.0])
                if event.key == pygame.K_LEFT:
                    self.command = self.command + np.array([0.1, 0.0, 0.0, 0.0])
                if event.key == pygame.K_RIGHT:
                    self.command = self.command + np.array([-0.1, 0.0, 0.0, 0.0])
                if event.key == pygame.K_w:
                    self.command = self.command + np.array([0.0, 0.0, 0.0, 1.0])
                if event.key == pygame.K_s:
                    self.command = self.command + np.array([0.0, 0.0, 0.0, -1.0])
                if event.key == pygame.K_a:
                    self.command = self.command + np.array([0.0, 0.0, 0.5, 0.0])
                if event.key == pygame.K_d:
                    self.command = self.command + np.array([0.0, 0.0, -0.5, 0.0])

        self.state, reward, terminal, _ = self.env.step(self.command)

    def fly(self):
        self.setup_pygame()

        frame = 1
        try:
            while True:
                self.get_states() # get current states
                # print(self.velocity)
                if frame == 5: # do optical flow on every 3rd frame
                    self.calc_optical_flow() # calculate optical flow
                    self.compute_R_and_t()
                    frame = 0
                    print('check')
                # self.plot_states()
                events = pygame.event.get() # get keyboard input
                self.send_commands(events) # send new command from keyboard input
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                # print(self.command)
                # print(self.velocity)
                # print(self.position)
                # print(self.body_vel)
                frame+=1
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    fly_uav = visual_odom()
    fly_uav.fly()
    # fly_uav.plot_states()
    print("Finished")
