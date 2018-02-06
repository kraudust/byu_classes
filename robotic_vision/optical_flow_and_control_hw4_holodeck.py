import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2
import scipy.io as sio
from copy import deepcopy
from  plot_holodeck_states import plot_states as plt_states
from pdb import set_trace as pause

def setup_pygame():
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("My Game")

def fly_uav():
    env = Holodeck.make("UrbanCity")
    # env = Holodeck.make("UrbanCity", Holodeck.GL_VERSION.OPENGL3)
    setup_pygame()
    state_dict = {'velocity':[],'orientation':[],'position':[],'imu':[]}
    # for i in range(10):
    #     env.reset()

    # roll, pitch, yaw rate, altitude positive axes:
    # x - out the back
    # y - out the right wing
    # z - out the top
    command = np.array([0.0, 0.0, 0.0, 0.0])
    state, reward, terminal, _ = env.step(command)

    # Create grid for optical flow
    grid = np.zeros((16*16,1,2))
    k = 0
    for j in range(16):
        for i in range(16):
            grid[k,0,0] = i*32 + 16
            grid[k,0,1] = j*32 + 16
            k+=1
    grid = np.float32(grid)

    # params for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Grab first image for optical flow
    image = state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

    # Convert to grayscale
    gray_old = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(image)

    j = 0
    for i in range(5000):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    command = command + np.array([0.0, -0.1, 0.0, 0.0])
                if event.key == pygame.K_DOWN:
                    command = command + np.array([0.0, 0.1, 0.0, 0.0])
                if event.key == pygame.K_LEFT:
                    command = command + np.array([0.1, 0.0, 0.0, 0.0])
                if event.key == pygame.K_RIGHT:
                    command = command + np.array([-0.1, 0.0, 0.0, 0.0])
                if event.key == pygame.K_w:
                    command = command + np.array([0.0, 0.0, 0.0, 5.0])
                if event.key == pygame.K_s:
                    command = command + np.array([0.0, 0.0, 0.0, -5.0])
                if event.key == pygame.K_a:
                    command = command + np.array([0.0, 0.0, 0.5, 0.0])
                if event.key == pygame.K_d:
                    command = command + np.array([0.0, 0.0, -0.5, 0.0])
                    
        state, reward, terminal, _ = env.step(command)

        # To access specific sensor data:
        velocity = state[Sensors.VELOCITY_SENSOR] # velocity
        orientation = state[Sensors.ORIENTATION_SENSOR] # orientation
        position = state[Sensors.LOCATION_SENSOR] # position
        imu = state[Sensors.IMU_SENSOR] # IMU
        image = state[Sensors.PRIMARY_PLAYER_CAMERA] # camera

        # Convert to grayscale
        gray_cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if j == 0:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_cur, grid, None, **lk_params)

            # Select points where optical flow exists (i.e. inverse existed)
            good_new = p1[st==1]
            good_old = grid[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 1)
                image = cv2.circle(image,(a,b),2,[0,0,255],-1)
            img = cv2.add(image,mask)
            cv2.imshow('frame',img)
            mask = np.zeros_like(image)
            gray_old = deepcopy(gray_cur)

        # cv2.imshow('Grayscale',gray_cur)
        cv2.imshow('Optic Flow', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Append sensor data to dictionary
        state_dict['velocity'].append(deepcopy(velocity))
        state_dict['orientation'].append(deepcopy(orientation))
        state_dict['position'].append(deepcopy(position))
        state_dict['imu'].append(deepcopy(imu))

        # Store the current image as the previous image for the next step
        # gray_old = deepcopy(gray_cur)

        # For a full list of sensors the UAV has, view the README
        print(command)
        print(i)
        j+=1
        if j == 2:
            j = 0
    sio.savemat('states.mat', state_dict)

if __name__ == "__main__":
    fly_uav()
    plt_states()
    print("Finished")
