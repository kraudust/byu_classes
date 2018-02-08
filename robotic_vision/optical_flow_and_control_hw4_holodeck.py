import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2
import scipy.io as sio
from copy import deepcopy
# from  plot_holodeck_states import plot_states as plt_states
from pdb import set_trace as pause

class holodeck_uav():
    def __init__(self):
        # positive axes
        # x - out the back
        # y - out the right wing
        # z - out the top
        # states
        self.phi = 0.0
        self.th = 0.0
        self.psi = 0.0
        self.velocity = None
        self.position = None
        self.psi_d = 0.9 # desired yaw angle
        self.h_d = 10.0 # desired height
        self.kp_psi = 5.0
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

        # params for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create grid for optical flow
        self.p0 = np.zeros((16*16,1,2))
        k = 0
        for j in range(16):
            for i in range(16):
                self.p0[k,0,0] = i*32 + 16
                self.p0[k,0,1] = j*32 + 16
                k+=1
        self.p0 = np.float32(self.p0)


    def setup_pygame(self):
        pygame.init()
        size = [200, 200]
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("My Game")

    def get_states(self):
        # To access specific sensor data:
        self.velocity = self.state[Sensors.VELOCITY_SENSOR] # velocity in world frame
        orientation = self.state[Sensors.ORIENTATION_SENSOR] # orientation (a rotation matrix)
        self.position = self.state[Sensors.LOCATION_SENSOR] # position in world frame
        # imu = self.state[Sensors.IMU_SENSOR] # IMU
        self.image = self.state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

        # Find orientation states
        self.th = np.arcsin(-orientation[2,0]) # pitch
        self.phi = np.arcsin(orientation[2,1]/np.cos(self.th)) # roll
        self.psi = np.arcsin(orientation[1,0]/np.cos(self.th)) # yaw

    def calc_optical_flow(self):

        # Convert to grayscale
        self.gray_cur = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        # Calculate optical flow
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0, None, **self.lk_params)

        # Select points where optical flow exists (i.e. inverse existed)
        good_new = self.p1[st==1]
        good_old = self.p0[st==1]

        # draw the tracks
        mask = np.zeros_like(self.image)
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 1)
            image = cv2.circle(self.image,(a,b),2,[0,0,255],-1)
        img = cv2.add(image,mask)
        self.gray_old = deepcopy(self.gray_cur)

        # cv2.imshow('Grayscale',gray_cur)
        cv2.imshow('Optic Flow', img)

    def send_commands(self,events):
        if self.flag == 0:
            self.command = np.array([0.0, 0.0, 0.0, self.h_d])
            self.command[2] = self.kp_psi*(self.psi_d - self.psi)
            self.state, reward, terminal, _ = self.env.step(self.command)
        elif self.flag == 1:
            # self.setup_pygame()
            print('leg')
            for event in events:
                print('face')
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
                        self.command = self.command + np.array([0.0, 0.0, 0.0, 5.0])
                    if event.key == pygame.K_s:
                        self.command = self.command + np.array([0.0, 0.0, 0.0, -5.0])
                    if event.key == pygame.K_a:
                        self.command = self.command + np.array([0.0, 0.0, 0.5, 0.0])
                    if event.key == pygame.K_d:
                        self.command = self.command + np.array([0.0, 0.0, -0.5, 0.0])
            self.state, reward, terminal, _ = self.env.step(self.command)

    def fly(self):
        # env = Holodeck.make("UrbanCity", Holodeck.GL_VERSION.OPENGL3)
        self.setup_pygame()
        # state_dict = {'velocity':[],'orientation':[],'position':[],'imu':[]}
        # pygame.init()
        # size = [200, 200]
        # screen = pygame.display.set_mode(size)
        # pygame.display.set_caption("My Game")

        for i in range(5000):
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.flag+=1

            self.get_states()
            self.send_commands(events)
            self.calc_optical_flow()
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # # Append sensor data to dictionary
            # state_dict['velocity'].append(deepcopy(velocity))
            # state_dict['orientation'].append(deepcopy(orientation))
            # state_dict['position'].append(deepcopy(position))
            # state_dict['imu'].append(deepcopy(imu))

            # Store the current image as the previous image for the next step
            # gray_old = deepcopy(gray_cur)

            # For a full list of sensors the UAV has, view the README
            # print(command)
            # print(velocity)

            print('phi: ', self.phi)
            print('th: ', self.th)
            print('psi: ', self.psi)
            print(self.position[2,0])
            print(i)
            print(self.flag)
            
        # sio.savemat('states.mat', state_dict)

if __name__ == "__main__":
    fly_uav = holodeck_uav()
    fly_uav.fly()
    print("Finished")
