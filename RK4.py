import numpy as np
import matplotlib.pyplot as plt


def Brusselator(a, b):
    def f(Y, t):
        x, y = Y
        return np.array([a + (x**2 * y) - ((b+1) * x), b * x - (x**2 * y)])
    return f


def RK4(N, tn, tau, f, x0, y0):
    Y0 = np.array([x0, y0])
    Yn = np.zeros((N+1, 2))
    Yn[0] = Y0

    for n in range(N):
        k1 = f(Yn[n], tn[n])
        k2 = f(Yn[n] + tau/2, tn[n] + tau/2*k1)
        k3 = f(Yn[n] + tau/2, tn[n] + tau/2*k2)
        k4 = f(Yn[n] + tau, tn[n] + tau*k3)
        Yn[n+1] = Yn[n] + tau/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Yn


def trajectoryPlotting():
    
    x = Y[:,0]
    y = Y[:,1]
    z = t_points
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='RK4')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    plt.title('Trajectory of concentrations (x,y)')
    plt.legend()
    plt.savefig("trajectoire.pdf")


def concentrationPlotting():

    plt.plot(t_points, Y[:,0], label='x RK4')
    plt.plot(t_points, Y[:,1], label='y RK4')
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


Y = RK4(N, t_points, tau_depart, f2, x0, y0)

concentrationPlotting()
trajectoryPlotting()