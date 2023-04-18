import numpy as np
import matplotlib.pyplot as plt


def Brusselator(a, b):
    def f(Y, t):
        x, y = Y
        return np.array([a + (x**2 * y) - ((b+1) * x), b * x - (x**2 * y)])
    return f


def RKF4Butcher(N, tn, tau, f, x0, y0):
    # Define the Butcher tableau coefficients
    beta = np.array([[1/4, 0, 0, 0, 0, 0], [3/32, 9/32, 0, 0, 0, 0], 
                     [1932/2197,-7200/2197, 7296/2197, 0, 0, 0], 
                     [439/216, -8, 3680/513, -845/4104, 0, 0], 
                     [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]])
    gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    alpha = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    
    Y0 = np.array([x0, y0])
    Yn = np.zeros((N+1, 2))
    Yn[0] = Y0

    # Calculate the intermediate stages
    for n in range(N):
        K = np.zeros((6, len(Yn[n])))
        K[0] = f(Yn[n], tn[n])
        for i in range(1, 6):
            temp = np.zeros_like(Yn[n])
            for j in range(i):
                temp += beta[i-1][j]*K[j]
            K[i] = f(Yn[n] + tau*temp, tn[n] + alpha[i]*tau)
        
        # Calculate the approximation at tn+tau
        Yn[n + 1] = Yn[n] + tau*np.dot(gamma, K)
    
    return Yn




def trajectoryPlotting():
    
    x = Y[:,0]
    y = Y[:,1]
    z = t_points
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='RKF4Butcher')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    plt.title('Trajectory of concentrations (x,y)')
    plt.legend()
    plt.savefig("trajectoire.pdf")


def concentrationPlotting():

    plt.plot(t_points, Y[:,0], label='x RKF4Butcher')
    plt.plot(t_points, Y[:,1], label='y RKF4Butcher')
    plt.xlabel('t')
    plt.legend()
    plt.title('Evolution temporelle des concentrations x et y')
    plt.grid()
    plt.savefig('evol-concentration.pdf')
    plt.show()

# Define the function f
(a,b)=(1,3)
(c,d)=(1,1.5)

f1 = Brusselator(a, b)
f2 = Brusselator(c, d)

# Define the parameters
T = 18
x0 = 0
y0 = 1
N = 1000
tau_depart = T/N
e_max = 0.0001

# Define the time points
t_points = np.linspace(0, T, N+1)

Y = RKF4Butcher(N, t_points, tau_depart, f1, x0, y0)
concentrationPlotting()
trajectoryPlotting()


