import matplotlib.pyplot as plt
import numpy as np
#from numpy.random import rand


J = 1
mu = 0.33
k = 1
B = 0


def initial_state(N):
    state = 2 * np.random.randint(2, size=(N, N, N)) - 1
    return state


def Metropolis_Loop(config, T):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                c = np.random.randint(0, N)
                s = config[a, b, c]
                ss = config[(a + 1) % N, b, c] + config[(a - 1) % N, b, c] + config[a, (b + 1) % N, c] + config[
                    a, (b - 1) % N, c] + config[a, b, (c + 1) % N] + config[a, b, (c - 1) % N]
                prod = 2 * s * ss
                if prod < 0:
                    s *= -1
                elif np.random.random() < np.exp(-float(prod) / (k * T)):
                    s *= -1
                config[a, b, c] = s
    return config


def Energy(config):
    energy1 = 0
    for i in range(len(config)):
        for j in range(len(config)):
            for k in range(len(config)):
                s = config[i, j, k]
                ss = config[(i + 1) % N, j, k] + config[(i - 1) % N, j, k] + config[i, (j + 1) % N, k] + config[
                    i, (j - 1) % N, k] + config[i, j, (k + 1) % N] + config[i, j, (k - 1) % N]
                energy1 += -J * s * ss
    return energy1 / 6.


def Magnetization(config):
    mag = np.sum(config)
    return mag


T_min = 0.1
T_max = 3.5
step = 90
interval = (T_max - T_min) / float(step)
N = 10
metropolis_step = 6000
calculation_step = 5000
n1 = 1. / (calculation_step * N * N * N)
n2 = 1. / (calculation_step * calculation_step * N * N * N)
T = np.linspace(T_min, T_max, step)
E = np.zeros(step)
M = np.zeros(step)
C = np.zeros(step)
X = np.zeros(step)

for tem in range(step):
    config = initial_state(N)

    En = 0
    Mag = 0
    En2 = 0
    Mag2 = 0

    for i in range(metropolis_step):
        Metropolis_Loop(config, T[tem])

    for i in range(calculation_step):
        Metropolis_Loop(config, T[tem])
        energy = Energy(config)
        magnetization = Magnetization(config)

        En = En + energy
        Mag = Mag + magnetization
        En2 = En2 + (energy * energy)
        Mag2 = Mag2 + (magnetization * magnetization)

        E[tem] = n1 * En
        M[tem] = n1 * Mag
        C[tem] = float((n1 * En2 - n2 * En * En)) / (k * T[tem] * T[tem])
        X[tem] = float((n1 * Mag2 - n2 * Mag * Mag)) / (k * T[tem])

plt.figure(figsize=(14.17, 7))
plt.subplot(2, 2, 1)
plt.scatter(T, E, s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title('Energy Vs. Temperature')

plt.subplot(2, 2, 2)
plt.scatter(T, abs(M), s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Magnetization Vs Temperature")

plt.subplot(2, 2, 3)
plt.scatter(T, C, s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Specific Heat")
plt.title("Specific Heat vs Temperature")

plt.subplot(2, 2, 4)
plt.scatter(T, abs(X), s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.title("Susceptibility vs Temperature")

plt.subplots_adjust(0.12, 0.11, 0.90, 0.81, 0.26, 0.56)
plt.suptitle("Simulation of 3D Ising Model by Metropolis Algorithm\n" + "Lattice Dimension:" + str(N) + "X" + str(
    N) + "X" + str(N) + "\n" + "External Magnetic Field(B)=" + str(B) + "\n" + "Metropolis Step=" + str(
    metropolis_step))
'''plt.savefig("ThreeD.png", dpi=500, facecolor='w', edgecolor='w',
            orientation='landscape', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)'''

plt.show()
