import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
import cv2

def setup_pygame():
    pygame.init()
    size = [200, 200]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("My Game")
    # pygame.mouse.set_visible(0)

def fly_uav():
    env = Holodeck.make("UrbanCity")
    setup_pygame()
    for i in range(10):
        env.reset()

        # roll, pitch, yaw rate, altitude positive axes:
        # x - out the back
        # y - out the right wing
        # z - out the top
        command = np.array([0.0, 0.0, 0.0, 20.0])
        for _ in range(5000):
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
            cv2.imshow('Original Image',image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # For a full list of sensors the UAV has, view the README
            print(command)

if __name__ == "__main__":
    fly_uav()
    print("Finished")
