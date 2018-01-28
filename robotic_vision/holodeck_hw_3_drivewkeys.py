import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

# This is a basic example of how to use the UAV agent
def uav_example():
    env = Holodeck.make("UrbanCity")

    for i in range(10):
        env.reset()

        # This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
        command = np.array([0, -0.2, 0.3, 20])
        for _ in range(800):
            state, reward, terminal, _ = env.step(command)

            # To access specific sensor data:
            velocity = state[Sensors.VELOCITY_SENSOR] # velocity
            orientation = state[Sensors.ORIENTATION_SENSOR] # orientation
            position = state[Sensors.LOCATION_SENSOR] # position
            imu = state[Sensors.IMU_SENSOR] # IMU
            # For a full list of sensors the UAV has, view the README
            print(velocity)

if __name__ == "__main__":
    uav_example()
    print("Finished")
