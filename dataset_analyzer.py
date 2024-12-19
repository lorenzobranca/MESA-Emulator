import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/lollo/binaries/mesaemulator/aligned_data'

# Load and preprocess data
time   = np.loadtxt(data_dir + '/time_grid_interp_aligned.dat')
mass   = np.load(data_dir + '/mass_interp_aligned.npy')
radius = np.load(data_dir + '/rad_interp_aligned.npy')

# Initial conditions
IC = np.loadtxt(data_dir + '/parameter_space.dat')[:, 1:]  # First column is constant (original mass)


for i in range(53):

    plt.plot(time, mass[i,:])

plt.xscale('log')
plt.show()
