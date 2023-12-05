import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the functions
def f(x, b1, b2):
    return np.maximum(0, np.maximum(0, x + b1) + b2)

def f2(x, b1, b2):
    return np.maximum(0, np.maximum(0, x + b2) + b1)

# Create a meshgrid for the inputs x, b1, b2
n = 1000
x = np.linspace(0, 20, n*4)
b1 = np.linspace(-2, 2, n)
b2 = np.linspace(-2, 2, n)
X, B1, B2 = np.meshgrid(x, b1, b2)

# Calculate function values
F = f(X, B1, B2)
F2 = f2(X, B1, B2)

# Flatten the arrays for plotting
X_flat = X.flatten()
B1_flat = B1.flatten()
B2_flat = B2.flatten()
F1_flat = F.flatten()
F2_flat = F2.flatten()



# Normalize F and F2 for color mapping
F1_norm = (F1_flat - F1_flat.min()) / (F1_flat.max() - F1_flat.min())
F2_norm = (F2_flat - F2_flat.min()) / (F2_flat.max() - F2_flat.min())

# Create the plot
fig = plt.figure(figsize=(22, 8))

# Scatter plots for both functions

alpha = .5

axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
Fs = [F1_flat, F2_flat]

for i, (F_flat, ax) in enumerate(zip(Fs, axs)):

    # Setting labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('B1 axis')
    ax.set_zlabel('B2 axis')
    ax.set_title('3D Scatter Plot of f(x, b1, b2) and f2(x, b1, b2)')

    # add data point
    a,b,c = (5, -1, 1)
    ax.scatter(a,b,c, c='black', s=60)

    # add 1st order taylor expansion of data point
    ax.plot([0,0], [0,b], [0,0], c='black')
    ax.plot([0,a], [b,b], [0,0], c='black')
    ax.plot([a,a], [b,b], [0,c], c='black')

    ax.plot([0,0], [0,0], [0,c], c='black')
    ax.plot([0,a], [0,0], [c,c], c='black')
    ax.plot([a,a], [0,b], [c,c], c='black')

    # add lines along x,y,z axis:
    n = 2
    A,B = np.linspace(-5, 5, n), np.zeros(n)
    ax.plot(A,B,B, 'r')
    ax.plot(B,A,B, 'g')
    ax.plot(B,B,A, 'b')

    n = 50000
    X_line = np.linspace(0, a, n)
    B1_line = np.zeros(n) + b
    B2_line = np.zeros(n)
    stacked_line = np.stack((X_line, B1_line, B2_line), axis=0)

    # Masks based on function value thresholds
    val, tol = .0, .001
    mask = (F_flat > val+1e8) & (F_flat < val+2*tol)
    
    # save results with limited mask
    stacked_plane = np.stack((X_flat[mask], B1_flat[mask], B2_flat[mask]), axis=0)

    # show values at higher func value
    val = 1
    mask |= (F_flat > val - tol) & (F_flat < val+tol)
    
    # plot
    # scatter = ax.scatter(X_flat[mask], B1_flat[mask], B2_flat[mask], s=.4*(20 + 30*F1_norm[mask]), alpha=alpha,
    #                      c=F_flat[mask], cmap='viridis'
    #                     #  c='gb'[i]
    #                      )
    # cbar = plt.colorbar(scatter, ax=ax)

    # plot point in line closest to function.
    dist = ((stacked_plane[:, :, None] - stacked_line[:, None, :])**2).sum(axis=0)**.5
    is_min = dist == dist.min()
    idx = np.argmin(np.sum(dist, axis=0))

    print(idx, stacked_line[:, idx])
    ax.scatter(*stacked_line[:, idx], c='red', s=90, alpha=1)
    ax.plot(*stacked_line, c='red')
    




# Show the plot
plt.show()
