import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_initial_configuration(N, random_influence=1):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    #A = random_influence * np.random.random((N, N))
    A = np.ones((N, N))*5
    A[0] = np.zeros(N)
    for i in range(N-1):
        A[i, 0] = 0
        A[i, N-1] = 0
    A[N-1] = np.zeros(N)
    return A

def update_function(matrix, N,  tau , delta):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    temp = np.zeros((N,N))
    for i in range(1, N - 1):
        for j in range(N - 1):
            temp[i][j] =  matrix[i,j] + tau * ( (matrix[i+1,j] - 2*matrix[i,j] + matrix[i-1,j])/ delta**2 + (matrix[i,j+1] - 2*matrix[i,j] + matrix[i,j - 1])/ delta**2  + matrix[i,j] - matrix[i,j]**3)
    return temp


def draw(A):
    """draw the concentrations"""
    c = plt.imshow(A, cmap='Greens',
                   interpolation='nearest', origin='lower')
    plt.colorbar(c)

    plt.title('reaction-diffusion function')
    plt.show()

def exercise(size, time_count):
    A = get_initial_configuration(size, 10)
    tau = 0.001
    delta = 0.1
    rez = [A[49][49]]
    draw(A)
    for i in range(time_count):
        A = update_function(A, size, tau, delta)
        rez.append(np.mean(A))
        if i % 100 == 0:
            draw(A)
    return np.array(rez)

def get_initial_configuration_1d(N, random_influence=1):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    A = random_influence * np.random.random(N)
    A[0] = 0
    A[N-1] = 0
    return A

def update_function_1d(matrix, N,  tau , delta):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    temp = np.zeros(N)
    for i in range(1, N - 1):
            temp[i] = matrix[i] + tau * ((matrix[i+1] - 2*matrix[i] + matrix[i-1]) / delta**2 + matrix[i] - matrix**3)
    return temp

def exercise_1d(size, time_count):
    A = get_initial_configuration_1d(size, 10)
    tau = 0.001
    delta = 0.1
    rez = [np.mean(A)]
    draw(A)
    for i in range(time_count):
        A = update_function(A, size, tau, delta)
        rez.append(np.mean(A))
        if i % 100 == 0:
            draw(A)
    return np.array(rez)