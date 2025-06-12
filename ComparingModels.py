import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pybamm

'''
models=[
    {
        "name": "SPM",
        "model_class": pybamm.lithium_ion.SPM,
        "time": [0, 3600],  # 1 hour
        "variables": ["Terminal voltage [V]", "Electrolyte current density [A.m-2]"]
    },
    {
        "name": "DFN",
        "model_class": pybamm.lithium_ion.DFN,
        "time": [0, 3600],  # 1 hour
        "variables": ["Terminal voltage [V]", "Electrolyte current density [A.m-2]"]
    },
    {
        "name": "SPMe",
        "model_class": pybamm.lithium_ion.SPMe,
        "time": [0, 3600],  # 1 hour
        "variables": ["Terminal voltage [V]", "Electrolyte current density [A.m-2]"]
    }

]
'''
# Function to create and solve a model

models=[
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.DFN(),
    pybamm.lithium_ion.SPMe()
]
solutions = []

for model in models:
    sim= pybamm.Simulation(model)
    solution= sim.solve([0, 3600])
    solutions.append(solution)


# Plotting results for each model
#pybamm.dynamic_plot(solutions)
pybamm.dynamic_plot(solutions, ["Terminal voltage [V]", "Electrolyte current density [A.m-2]"])


