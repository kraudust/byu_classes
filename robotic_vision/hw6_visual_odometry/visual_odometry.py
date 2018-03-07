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
        self.command = np.array([0.0, 0.0, 0.0, 0.0]) # roll, pitch, yaw rate, altitude
        self.state, reward, terminal, _ = self.env.step(self.command)

        # Grab first image for optical flow
        self.image = self.state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

        # Convert to grayscale
        self.gray_old = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        self.gray_cur = None # current grascale image for optical flow

        # Grab first set of features to track
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
        self.p0 = cv2.goodFeaturesToTrack(self.gray_old, mask = None, **self.feature_params)
        self.p1 = None

        # params for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

    def calc_optical_flow(self):
        # Calculate new points that grids have moved to
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0, None, **self.lk_params)

        # Select points where optical flow exists (i.e. inverse existed)
        good_new = self.p1[st==1]
        good_old = self.p0[st==1]

        # draw the tracks for visualization
        mask = np.zeros_like(self.image)
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 1)
            image = cv2.circle(self.image,(a,b),2,[0,0,255],-1)
        img = cv2.add(image,mask)

        # cv2.imshow('Optic Flow', img)
        self.gray_old = deepcopy(self.gray_cur)

        # Calculate new features to track in current image
        self.p0 = cv2.goodFeaturesToTrack(self.gray_old, mask = None, **self.feature_params)

    def plot_states(self):
        if self.init_plots == True:
            self.app = pg.QtGui.QApplication([])
            self.ekfplotwin = pg.GraphicsWindow(size=(800,400))
            self.ekfplotwin.setWindowTitle('Position')
            self.ekfplotwin.setInteractive(True)
            self.plots = self.ekfplotwin.addPlot(1,1)
            self.xycurves = self.plots.plot(pen=(0,0,255))
            self.xdata = []
            self.ydata = []
            self.init_plots = False
        else:
            self.xdata.append(self.position[0,0])
            self.ydata.append(self.position[1,0])
            self.xycurves.setData(self.xdata, self.ydata)
        self.app.processEvents()


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
                if frame == 3: # do optical flow on every 3rd frame
                    self.calc_optical_flow() # calculate optical flow
                    frame = 0
                self.plot_states()
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
    fly_uav.plot_states()
    print("Finished")
