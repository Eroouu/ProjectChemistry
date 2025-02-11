import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

A = np.ones((3, 3))
A[1, 1] = 0

right_neighbor = np.roll(A,  # the matrix to permute
                         (0, -1),  # we want the right neighbor, so we shift the whole matrix -1 in the x-direction)
                         (0, 1)  # apply this in directions (y,x)
                        )

def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""

    laplace = -4*M
    laplace += np.roll(M, (0, -1), (0, 1))  # right neighbor
    laplace += np.roll(M, (0, +1), (0, 1))  # left neighbor
    laplace += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    laplace += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return laplace

def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """

    # Let's get the discrete Laplacians first
    laplace_a = discrete_laplacian(A)
    laplace_b = discrete_laplacian(B)

    # Now apply the update formula
    diff_a = (DA * laplace_a - A * B**2 + f * (1-A)) * delta_t
    diff_b = (DB * laplace_b + A * B**2 - (k + f) *B) * delta_t

    A += diff_a
    B += diff_b

    return A, B

def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """

    # We start with a configuration where on every grid cell
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N, N)) + random_influence * np.random.random((N,N))

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N, N))

    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/ 10.0)

    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B

def draw(A,B):
    """draw the concentrations"""
    fig, ax = pl.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')

# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.16
DB = 0.08

# define feed/kill rates
f = 0.060
k = 0.062

# grid size
N = 200

# simulation steps
N_simulation_steps = 10000

def example1():
    DA, DB, f, k = 0.14, 0.06, 0.035, 0.065  # bacteria
    A, B = get_initial_configuration(200)
    temp = [A[20, 20]]
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
        temp.append(np.mean(A)+np.mean(B))

    draw(A, B)
    return np.array(temp)

def example2():
    DA, DB, f, k = 0.16, 0.08, 0.060, 0.062  # bacteria
    A, B = get_initial_configuration(200)
    temp = [A[20, 20]]
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
        temp.append(np.mean(A)+np.mean(B))

    draw(A, B)
    return np.array(temp)
