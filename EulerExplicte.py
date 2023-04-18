import numpy as np
import matplotlib.pyplot as plt

def Brusselator(a, b):
    def f(Y, t):
        x, y = Y
        return np.array([a + (x**2 * y) - ((b+1) * x), b * x - (x**2 * y)])
    return f

def EulerExplicit(f, Y0, t0, T, N):
    # Initialisation
    Y = np.zeros((N+1, len(Y0)))
    Y[0,:] = Y0
    t = np.linspace(t0, T, N+1)
    tau = t[1] - t[0]
    
    # Boucle pour calculer les approximations successives de Y
    for n in range(N):
        Y[n+1,:] = Y[n,:] + tau*f(Y[n,:], t[n])
        
    return Y, t

def trajectoryPlotting():
    x = Y[:,0]
    y = Y[:,1]
    z = t_points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='EulerExplicit')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    plt.title('Trajectory of concentrations (x,y)')
    plt.legend()
    plt.savefig("trajectoire.pdf")
    
    plt.show()



def concentrationPlotting():
    plt.plot(t, Y[:,0], label='x')
    plt.plot(t, Y[:,1], label='y')
    plt.xlabel('t')
    plt.legend()
    plt.title('Evolution temporelle des concentrations x et y')
    plt.grid()
    plt.savefig('evol-concentration.pdf')
    plt.show()


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

#Define the initial condition
Y0 = np.array([x0, y0])

# Define the time points
t_points = np.linspace(0, T, N+1)
# Résolution du système d'équations différentielles avec le schéma d'Euler explicite
#Cas 1
Y, t = EulerExplicit(f1, Y0, 0, T, N)
# Traçage des courbes
concentrationPlotting()
trajectoryPlotting()

#Cas 2
Y, t = EulerExplicit(f2, Y0, 0, T, N)
# Traçage des courbes
concentrationPlotting()
trajectoryPlotting()


