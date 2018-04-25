import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as pause

class rls():
    def __init__(self, type_fit, initial_guess):
        if type_fit == 'linear':
            self.n = 2
            self.est_var = np.matrix(np.zeros((self.n,1))) # estimated variables
            self.est_var[0,0] = initial_guess[0]
            self.est_var[1,0] = initial_guess[1]
        elif type_fit == 'quadratic':
            self.n = 3
            self.est_var = np.matrix(np.zeros((self.n,1))) # estimated variables
            self.est_var[0,0] = initial_guess[0]
            self.est_var[1,0] = initial_guess[1]
            self.est_var[2,0] = initial_guess[2]
        self.type_fit = type_fit
        self.data = [] # a list of the data points
        self.Pm = np.matrix(np.eye(self.n))

    def update_variables(self, data_point): # function to update the variable estimates with new data_point
        if self.type_fit == 'quadratic':
            am1 = np.matrix(np.array([[data_point[0]**2], [data_point[0]], [1]]))
        elif self.type_fit == 'linear':
            am1 = np.matrix(np.array([[data_point[0]], [1]]))
        # if len(self.data) < 3:
        #     self.data.append(data_point)
        bm1 = data_point[1]
        Km1 = (self.Pm*am1)/(1. + am1.T*self.Pm*am1)
        self.est_var = self.est_var + Km1*(bm1 - am1.T*self.est_var)
        self.Pm = self.Pm - Km1*am1.T*self.Pm

if __name__=='__main__':
    x = np.linspace(10,0.76,45)
    a = -0.01176
    b = 0.1265
    # a = 0.05
    # b = -0.5
    c = 0.4118
    # m = 2
    # b = 3
    z = a*x**2 + b*x + c + .05*(np.random.rand(len(x)) - 0.5)
    y = a*x + b + .05*(np.random.rand(len(x)) - 0.5)
    # var_store = np.matrix(np.zeros((2,len(x))))
    var_store = np.matrix(np.zeros((3,len(x))))

    # fit = rls('linear', np.array((0.05, -0.5)))
    fit = rls('quadratic', np.array((0, 0.1, 0.4)))
    for i in range(len(x)):
        fit.update_variables(np.array([x[i], z[i]]))
        # fit.update_variables(np.array([x[i], y[i]]))
        var_store[:,i] = fit.est_var

    print(fit.est_var)

    plt.figure()
    # plt.plot(x,y)
    plt.plot(x,z)
    # plt.plot(x,np.squeeze(np.array(var_store[0,:])*x) + np.squeeze(np.array(var_store[1,:])))
    plt.plot(x,np.squeeze(np.array(var_store[0,:])*x**2) + np.squeeze(np.array(var_store[1,:])*x) + np.squeeze(np.array(var_store[2,:])))

    plt.figure()
    plt.plot(x,np.squeeze(np.array(var_store[0,:])))
    plt.plot(x,np.squeeze(np.array(var_store[1,:])))
    plt.plot(x,np.squeeze(np.array(var_store[2,:])))

    plt.show()


