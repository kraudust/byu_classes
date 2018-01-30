import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from transformations import euler_from_matrix

def plot_states():
    states = sio.loadmat('states.mat')
    num_entries = np.shape(states["orientation"])[0]
    # print states.keys()
    # print np.shape(states["orientation"])
    # print np.shape(states["orientation"][0,:,:])
    # print type(states["orientation"][0,:,:])
    # R = np.matrix(states["orientation"][0,:,:])
    x = states["position"][:,0,0]
    y = states["position"][:,1,0]
    z = states["position"][:,2,0]
    R = np.empty([3,num_entries])
    for i in range(num_entries):
        R[:,i] = euler_from_matrix(np.matrix(states["orientation"][i,:,:]), axes='sxyz')

    phi = R[0,:]
    th = R[1,:]
    psi = R[2,:]

    plt.figure()
    plt.suptitle('Position States')
    plt.subplot(311)
    plt.plot(x)
    plt.ylabel('x')
    plt.subplot(312)
    plt.plot(y)
    plt.ylabel('y')
    plt.subplot(313)
    plt.plot(z)
    plt.ylabel('z')
    plt.xlabel('Time')

    plt.figure()
    plt.suptitle('Orientation States')
    plt.subplot(311)
    plt.plot(phi)
    plt.ylabel('phi')
    plt.subplot(312)
    plt.plot(th)
    plt.ylabel('theta')
    plt.subplot(313)
    plt.plot(psi)
    plt.ylabel('psi')
    plt.xlabel('Time')

    plt.show()
if __name__ == "__main__":
    plot_states()

