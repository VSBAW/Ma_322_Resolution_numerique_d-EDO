# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:23:48 2023

@author: Valen
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt 


def Brusselator(a,b) : 
    def f(Y): 
        x = Y[0]
        y = Y[1]
        Yprimx = a + (x**2)*y - (b+1)*x
        Yprimy = b*x - (x**2)*y
        return np.array([Yprimx, Yprimy])
    return f




def RKF4Butcher(f, Y0, t0, T, N):
    """
    """
    tau = (T - t0) / N

    gamma = np.array([16/135, 0, 6656/12825, 25861/56430, -9/50, 2/55])
    beta = np.array([[0, 0, 0, 0, 0, 0],
                      [1/4, 0, 0, 0, 0, 0],
                      [3/32, 9/32, 0, 0, 0, 0],
                      [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                      [439/216, -8, 3680/513, -845/4104, 0, 0],
                      [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]])
    
    Yn = np.zeros((N+1, 2))
    Yn[0] = Y0

    for i in range(N):
        K1 = f(Yn[i])
        K2 = f(Yn[i] + tau*beta[1, 0]*K1)
        K3 = f(Yn[i] + tau*(beta[2, 0]*K1 + beta[2, 1]*K2))
        K4 = f(Yn[i] + tau*(beta[3, 0]*K1 + beta[3, 1]*K2 + beta[3, 2]*K3))
        K5 = f(Yn[i] + tau*(beta[4, 0]*K1 + beta[4, 1]*K2 + beta[4, 2]*K3 + beta[4, 3]*K4))
        K6 = f(Yn[i] + tau*(beta[5, 0]*K1 + beta[5, 1]*K2 + beta[5, 2]*K3 + beta[5, 3]*K4 + beta[5, 4]*K5))
        
        Yn[i+1] = Yn[i] + tau*(gamma[0]*K1 + gamma[1]*K2 + gamma[2]*K3 + gamma[3]*K4 + gamma[4]*K5 + gamma[5]*K6)

    return Yn

def ode_RK4(f, a, b,ic, N):
    h = (b - a) / N      
    Lt = np.linspace(a, b, N) 
    Ly = np.empty((N, np.size(ic)),dtype = float) 
    Ly[0,:] = ic
    for i in range(N-1):
        
        k1 = h*f(Lt[i], Ly[i,:])
        y1 = Ly[i,:] + 1/2*k1
        k2 = h* f(Lt[i]+h/2, y1)
        y2 = Ly[i,:] + 1/2*k2
        k3 = h* f(Lt[i]+h/2,y2) 
        y3 = Ly[i,:] + k3
        k4 = h* f(Lt[i]+h, y3)
        k = (k1+2*k2+2*k3+k4)/6
        Ly[i+1,:] = Ly[i,:] + k
    return (Lt, Ly)





def RK45(f,Y0,t,tau):
    tableau = np.array([
                        [0,         0,          0,          0,          0,      0],
                        [1/4,       0,          0,          0,          0,      0],
                        [3/32,      9/32,       0,          0,          0,      0],
                        [1932/2197, -7200/2197, 7296/2197,  0,          0,      0],
                        [439/216,   -8,         3680/513,   -845/4104,  0,      0],
                        [-8/27,     2,          -3544/2565, 1859/4104,  -11/40, 0]
                        ])
    
    gamma = np.array([16/135,0,6666/12825,28561/56430,(-9/50),2/55])
    gamma_barre = np.array([25/216,0,1408/2565,2197/4104,(-1/5),0])

    K1 = f(Y0)
    K2 = f(Y0 + tau*tableau[1, 1]*K1)
    K3 = f(Y0 + tau*(tableau[2, 1]*K1 + tableau[2, 2]*K2))
    K4 = f(Y0 + tau*(tableau[3, 1]*K1 + tableau[3, 2]*K2 + tableau[3, 3]*K3))
    K5 = f(Y0 + tau*(tableau[4, 1]*K1 + tableau[4, 2]*K2 + tableau[4, 3]*K3 + tableau[4, 4]*K4))
    K6 = f(Y0 + tau*(tableau[5, 1]*K1 + tableau[5, 2]*K2 + tableau[5, 3]*K3 + tableau[5, 4]*K4 + tableau[5, 5]*K5))

    Yn = Y0 + tau*(gamma[0]*K1 + gamma[2]*K3 + gamma[3]*K4 + gamma[4]*K5 + gamma[5]*K6)
    Zn = Y0 + tau*(gamma_barre[0]*K1 + gamma_barre[2]*K3 + gamma_barre[3]*K4 + gamma_barre[4]*K5 + gamma_barre[5]*K6)
    
    return Yn, Zn
    
def stepRK45(f, Y0, tau_ini, epsilon_max):
    dt = tau_ini
    epsilon = np.inf
    while epsilon > epsilon_max:
        Yn, Zn = RK45(f, Y0, tau_ini, dt)
        epsilon = np.linalg.norm(Zn - Yn)
        e = 0.9*((epsilon_max/epsilon)**(1/5))
        if e < 0.1:
            dt = 0.1 * dt
        elif e > 5:
            dt = 5 * dt
        else:
            dt = e *dt
    tau_next = dt
    return Yn, tau_next


def trajRK45(f, Y0, t0, T, N, epsilon_max):

    Yi, ti = Y0, t0
    temps = ti
    tpas = [t0]
    Y = [Y0]
    dt = (T-t0)/N
    t = [temps]
    while temps <= T:        
        Ynext, taunext = stepRK45(f, Yi, dt, epsilon_max)
        Y.append(Ynext)
        Yi = Ynext
        dt = taunext
        temps += taunext
        t.append(temps)
        tpas.append(dt)
        if temps>T:
            taunext=temps-T
    print(t)
    return np.array(tpas), np.array(Y), np.array(t)

def trajectoryPlotting():
    # Define the function f
    (a,b)=(1,3)
    (c,d)=(1,1.5)
    f = Brusselator(c, d)
    f2 = Brusselator(a, b)
    
    # Define the parameters
    T = 18
    x0 = 0
    y0 = 1
    N = 1000
    tau_depart = T/N
    e_max = 0.0001
    t0 = 0
    # Define the initial condition
    Y0 = np.array([x0, y0])

    # Define the time points
    t_points = np.linspace(0, T, N+1)


    Y = RKF4Butcher(f2, Y0, t0, T, N)
    liste_tau, Y2, liste_temps  = trajRK45(f2, Y0, t0, T, N, e_max)


    x = Y[:,0]
    y = Y[:,1]
    z = t_points

    x2 = Y2[:,0]
    y2 = Y2[:,1]
    z2 = liste_temps
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='RK4')
    ax.plot(x2, y2, z2, label='stepRK45')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    plt.title('Trajectory of concentrations (x,y)')
    plt.legend()
    plt.show()
    print(liste_temps)
    plt.plot(liste_temps, liste_tau)

    plt.xlabel('temps')
    plt.ylabel('pas')
    plt.title('variation de la valeur des pas de temps en fonction du temps')
    plt.show()

def concentrationPlotting():
    # Define the function f
    (a,b)=(1,3)
    (c,d)=(1,1.5)
    f = Brusselator(c, d)
    f2 = Brusselator(a, b)
    
    # Define the parameters
    T = 18
    x0 = 0
    y0 = 1
    N = 1000
    #tau_depart = T/N
    tau_ini= T/N
    e_max = 0.0001
    epsilon_max=10**(-3)
    t0 = 0
    # Define the initial condition
    Y0 = np.array([x0, y0])

    # Define the time points
    t_points = np.linspace(0, T, N+1)
    
    Y = RKF4Butcher(f, Y0, t0, T, N)
    liste_tau, Y2, liste_temps  = trajRK45(f2, Y0, t0, T, N, e_max)
    

    plt.plot(liste_temps, Y2[:,0], label='x RK45')
    plt.plot(liste_temps, Y2[:,1], label='y RK45')
    plt.xlabel('t')
    plt.legend()
    plt.title('Evolution temporelle des concentrations x et y')
    plt.grid()
    plt.savefig('evol-concentration.pdf')
    plt.show()

trajectoryPlotting()
concentrationPlotting()