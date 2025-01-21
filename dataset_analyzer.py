import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data_dir = '/home/lollo/binaries/mesaemulator/aligned_data'

# Load and preprocess data
#time   = np.loadtxt(data_dir + '/time_grid_interp_aligned.dat')
#mass   = np.load(data_dir + '/mass_interp_aligned.npy')
#radius = np.load(data_dir + '/rad_interp_aligned.npy')

# Initial conditions
#IC = np.loadtxt(data_dir + '/parameter_space.dat')[:, 1:]  # First column is constant (original mass)

#here build a funcion to map the age

data_row = pd.read_parquet('mesa_data_row/data.parquet')

q   = np.zeros(53)
ai  = np.zeros(53)
age = np.zeros(53)
for i in range(53):

    print(data_row["q"][i], data_row["ai"][i], data_row["age"][i][-1])
    age[i] = data_row["age"][i][-1]
    q[i]   = data_row["q"][i]
    ai[i]  = data_row["ai"][i]

from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

X = np.vstack((q, ai))
y = age

model.fit(np.transpose(X), y)

