import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2
import scipy.io as sio
from copy import deepcopy
from  plot_holodeck_states import plot_states as plt_states

def setup_pygame():
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("My Game")
    # pygame.mouse.set_visible(0)

def fly_uav():
    # env = Holodeck.make("UrbanCity")
    env = Holodeck.make("UrbanCity", Holodeck.GL_VERSION.OPENGL3)
    setup_pygame()
    state_dict = {'velocity':[],'orientation':[],'position':[],'imu':[]}
    # for i in range(10):
    #     env.reset()

    # roll, pitch, yaw rate, altitude positive axes:
    # x - out the back
    # y - out the right wing
    # z - out the top
    command = np.array([0.0, 0.0, 0.0, 0.0])
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
                    command = command + np.array([0.0, 0.0, 0.0, 1.0])
                if event.key == pygame.K_s:
                    command = command + np.array([0.0, 0.0, 0.0, -1.0])
                if event.key == pygame.K_a:
                    command = command + np.array([0.0, 0.0, 0.1, 0.0])
                if event.key == pygame.K_d:
                    command = command + np.array([0.0, 0.0, -0.1, 0.0])
                    
        state, reward, terminal, _ = env.step(command)

        # To access specific sensor data:
        velocity = state[Sensors.VELOCITY_SENSOR] # velocity
        orientation = state[Sensors.ORIENTATION_SENSOR] # orientation
        position = state[Sensors.LOCATION_SENSOR] # position
        imu = state[Sensors.IMU_SENSOR] # IMU
        image = state[Sensors.PRIMARY_PLAYER_CAMERA]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # Bluring filter
        # blur = cv2.blur(frame,(5,5))
        # blur_gray = cv2.blur(gray,(5,5))
        # Canny edge detection
        # canny = cv2.Canny(frame,100,200)
        canny_gray = cv2.Canny(gray,100,200)
        # canny_blur = cv2.Canny(blur,100,200)
        cv2.imshow('Canny Edge Detection',canny_gray)
        cv2.imshow('Grayscale',gray)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Append sensor data to dictionary
        state_dict['velocity'].append(deepcopy(velocity))
        state_dict['orientation'].append(deepcopy(orientation))
        state_dict['position'].append(deepcopy(position))
        state_dict['imu'].append(deepcopy(imu))

        # For a full list of sensors the UAV has, view the README
        print(command)
        print(i)
    sio.savemat('states.mat', state_dict)

if __name__ == "__main__":
    fly_uav()
    plt_states()
    print("Finished")
