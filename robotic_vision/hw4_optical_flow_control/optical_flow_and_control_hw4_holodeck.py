import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2
import scipy.io as sio
from copy import deepcopy
import matplotlib.pyplot as plt
# from  plot_holodeck_states import plot_states as plt_states
from pdb import set_trace as pause

class holodeck_uav():
    def __init__(self):
        # positive axes
        # x - out the back
        # y - out the right wing
        # z - out the top
        # states
        self.orientation = None
        self.phi = 0.0
        self.phi_max = 10.0 * np.pi/180.0
        self.th = 0.0
        self.psi = 0.0
        self.velocity = None
        self.body_vel = None
        self.position = None
        # self.psi_d = 0.9 # desired yaw angle
        self.psi_d = 0.55 # desired yaw angle
        self.h_d = 2.0 # desired height
        self.kp_psi = 0.8
        self.k_roll = 3.0
        self.k_obst = 0.12
        self.k_height = 0.5
        self.counter = 0
        self.state = {} # states dictionary containing velocity, orientation, position, and imu
        self.flag = 0
        self.of_l = [] #optic flow
        self.of_r = [] #optic flow
        self.of_ml = [] #optic flow
        self.of_mr = [] #optic flow
        self.of_m = [] #optic flow
        self.of_b = [] #optic flow
        self.xl = None
        self.xr = None
        self.yl = None
        self.yr = None
        self.tot = None
        self.bot = None
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

        # Create grids for optical flow ---------------------------------------------------------------
        # main grid
        self.p0 = np.zeros((16*16,1,2))
        k = 0
        for j in range(16):
            for i in range(16):
                self.p0[k,0,0] = i*32 + 16
                self.p0[k,0,1] = j*32 + 16
                k+=1
        self.p0 = np.float32(self.p0) # the full grid

        # left grid
        self.p0_l = np.zeros((16*5,1,2))
        k = 0
        for j in range(16):
            for i in range(5):
                self.p0_l[k,0,0] = i*32 + 16
                self.p0_l[k,0,1] = j*32 + 16
                k+=1
        self.p0_l = np.float32(self.p0_l)
        # pause()

        # right grid
        self.p0_r = np.zeros((16*5,1,2))
        k = 0
        for j in range(16):
            for i in range(5):
                self.p0_r[k,0,0] = i*32 + 16 * 23
                self.p0_r[k,0,1] = j*32 + 16
                k+=1
        self.p0_r = np.float32(self.p0_r)

        # middle left grid
        self.p0_ml = np.zeros((8*8,1,2))
        k = 0
        for j in range(8):
            for i in range(8):
                self.p0_ml[k,0,0] = i*32 + 16
                self.p0_ml[k,0,1] = j*32 + 16 * 9
                k+=1
        self.p0_ml = np.float32(self.p0_ml)

        # middle right grid
        self.p0_mr = np.zeros((8*8,1,2))
        k = 0
        for j in range(8):
            for i in range(8):
                self.p0_mr[k,0,0] = i*32 + 16 * 16
                self.p0_mr[k,0,1] = j*32 + 16 * 9
                k+=1
        self.p0_mr = np.float32(self.p0_mr)

        # middle grid
        self.p0_m = np.zeros((8*8,1,2))
        k = 0
        for j in range(8):
            for i in range(8):
                self.p0_m[k,0,0] = i*32 + 16 * 8 
                self.p0_m[k,0,1] = j*32 + 16 * 9
                k+=1
        self.p0_m = np.float32(self.p0_m)

        # bottom grid
        self.p0_b = np.zeros((5*5,1,2))
        k = 0
        for j in range(5):
            for i in range(5):
                self.p0_b[k,0,0] = i*32 + 16 * 13
                self.p0_b[k,0,1] = j*32 + 16 * 23
                k+=1
        self.p0_b = np.float32(self.p0_b)
        # ---------------------------------------------------------------------------------------------

    def setup_pygame(self):
        pygame.init()
        size = [200, 200]
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("My Game")

    def plot_grid(self,grid):
        plt.scatter(grid[:,0,0], grid[:,0,1])
        plt.show()

    def get_states(self):
        # To access specific sensor data:
        self.velocity = self.state[Sensors.VELOCITY_SENSOR] # velocity in world frame
        self.orientation = self.state[Sensors.ORIENTATION_SENSOR] # orientation (a rotation matrix)
        self.position = self.state[Sensors.LOCATION_SENSOR] # position in world frame
        # imu = self.state[Sensors.IMU_SENSOR] # IMU
        self.image = self.state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

        # Find orientation states
        self.th = np.arcsin(-self.orientation[2,0]) # pitch
        self.phi = np.arcsin(self.orientation[2,1]/np.cos(self.th)) # roll
        self.psi = np.arcsin(self.orientation[1,0]/np.cos(self.th)) # yaw

    def calc_optical_flow(self):
        # Convert to grayscale
        self.gray_cur = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        # Calculate new points that grids have moved to
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0, None, **self.lk_params)
        # self.p1_l, st_l, err_l = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_l, None, **self.lk_params)
        # self.p1_r, st_r, err_r = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_r, None, **self.lk_params)
        self.p1_ml, st_ml, err_ml = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_ml, None, **self.lk_params)
        self.p1_mr, st_mr, err_mr = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_mr, None, **self.lk_params)
        self.p1_m, st_m, err_m = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_m, None, **self.lk_params)
        self.p1_b, st_b, err_b = cv2.calcOpticalFlowPyrLK(self.gray_old, self.gray_cur, self.p0_b, None, **self.lk_params)

        # Select points where optical flow exists (i.e. inverse existed)
        good_new = self.p1[st==1]
        good_old = self.p0[st==1]
        # good_new_l = self.p1_l[st_l==1]
        # good_old_l = self.p0_l[st_l==1]
        # good_new_r = self.p1_r[st_r==1]
        # good_old_r = self.p0_r[st_r==1]
        good_new_ml = self.p1_ml[st_ml==1]
        good_old_ml = self.p0_ml[st_ml==1]
        good_new_mr = self.p1_mr[st_mr==1]
        good_old_mr = self.p0_mr[st_mr==1]
        good_new_m = self.p1_m[st_m==1]
        good_old_m = self.p0_m[st_m==1]
        good_new_b = self.p1_b[st_b==1]
        good_old_b = self.p0_b[st_b==1]

        # draw the tracks for visualization
        mask = np.zeros_like(self.image)
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 1)
            image = cv2.circle(self.image,(a,b),2,[0,0,255],-1)
        img = cv2.add(image,mask)

        # Calculate optic flow using old and new points
        # self.of_l = []
        # for i,(new,old) in enumerate(zip(good_new_l,good_old_l)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     self.of_l.append([a-c, b-d])
        # self.of_l = np.transpose(self.of_l).tolist()
        # self.of_r = []
        # for i,(new,old) in enumerate(zip(good_new_r,good_old_r)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     self.of_r.append([a-c, b-d])
        # self.of_r = np.transpose(self.of_l).tolist()
        self.of_ml = []
        for i,(new,old) in enumerate(zip(good_new_ml,good_old_ml)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.of_ml.append([a-c, b-d])
        self.of_ml = np.transpose(self.of_ml).tolist()
        self.of_mr = []
        for i,(new,old) in enumerate(zip(good_new_mr,good_old_mr)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.of_mr.append([a-c, b-d])
        self.of_mr = np.transpose(self.of_mr).tolist()
        self.of_m = []
        for i,(new,old) in enumerate(zip(good_new_m,good_old_m)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.of_m.append([a-c, b-d])
        self.of_m = np.transpose(self.of_m).tolist()
        self.of_b = []
        for i,(new,old) in enumerate(zip(good_new_b,good_old_b)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.of_b.append([a-c, b-d])
        self.of_b = np.transpose(self.of_b).tolist()

        cv2.imshow('Optic Flow', img)
        self.gray_old = deepcopy(self.gray_cur)

    def calc_and_send_commands(self,events):
        if self.flag == 0:
            self.command = np.array([0.0, 0.0, 0.0, self.h_d])
            self.command[2] = self.kp_psi*(self.psi_d - self.psi)
            print('Moving to initial desired position')
        elif self.flag == 1:
            print('Manual Control')
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
                        self.command = self.command + np.array([0.0, 0.0, 0.1, 0.0])
                    if event.key == pygame.K_d:
                        self.command = self.command + np.array([0.0, 0.0, -0.1, 0.0])
        elif self.flag == 2:
            print('Canyon Following')
            if abs(self.command[0]) < self.phi_max:
                self.command[0] = (self.xl + self.xr + self.body_vel[1])[0]/50.0*self.k_roll
            else:
                if self.command[0] < 0:
                    self.command[0] = -self.phi_max + 0.01
                else:
                    self.command[0] = self.phi_max - 0.01
            # Obstacle avoidance and centering
            self.command[2] = (self.yr - self.yl)*self.k_obst + (self.psi_d - self.psi)*self.kp_psi*3.0

            # Height centering
            self.command[3] = self.h_d + self.bot*self.k_height

        elif self.flag == 3: # emergency stop
            print('STOP!!!! Obstacle directly in front!')
            if self.counter < 30:
                self.command[1] = 1.2
                self.counter += 1
            elif self.counter >= 30 and self.counter < 40:
                self.command[1] = -0.1
                self.counter += 1
            elif self.counter >= 40 and self.counter <=80:
                self.command[1] = 0
                self.counter += 1
            else:
                self.flag = 1
                self.counter = 0

        self.state, reward, terminal, _ = self.env.step(self.command)

    def fly(self):
        # env = Holodeck.make("UrbanCity", Holodeck.GL_VERSION.OPENGL3)
        self.setup_pygame()
        # self.plot_grid(self.p0_l)
        # pause()
        # state_dict = {'velocity':[],'orientation':[],'position':[],'imu':[]}
        # pygame.init()
        # size = [200, 200]
        # screen = pygame.display.set_mode(size)
        # pygame.display.set_caption("My Game")

        try:
            while True:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            th_d = np.arcsin(self.orientation[2][0] * -1.0)
                            self.psi_d = np.arcsin(self.orientation[1][0]) / np.cos(th_d)
                            self.flag+=1

                self.get_states()
                self.calc_optical_flow()

                # Do optic flow averaging calculations for flight control
                # ysuml,ysumr = np.average(np.abs(self.of_l[1])),np.average(np.abs(self.of_r[1]))
                # xavgl,xavgr = np.average(np.abs(self.of_l[0])),np.average(np.abs(self.of_r[0]))
                self.xl,self.xr = np.average(self.of_ml[0]),np.average(self.of_mr[0])
                self.yl,self.yr = np.average(self.of_ml[1]),np.average(self.of_mr[1])
                self.tot = np.average(np.abs(self.of_m[0]))+np.average(np.abs(self.of_m[1]))
                self.bot = np.average(self.of_b[1])

                # Body Frame Velocities
                self.body_vel = np.matmul(self.orientation, np.divide(self.velocity, 100.0))
                print(self.tot)

                if self.tot > 3.5:
                    self.flag = 3

                self.calc_and_send_commands(events)
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

                # print('phi: ', self.phi)
                # print('th: ', self.th)
                # print('psi: ', self.psi)
                # print('height: ', self.position[2,0]/100.0)
                print(self.command)
                print('flag: ', self.flag)
        except KeyboardInterrupt:
            pass
            
        # sio.savemat('states.mat', state_dict)

if __name__ == "__main__":
    fly_uav = holodeck_uav()
    fly_uav.fly()
    print("Finished")
