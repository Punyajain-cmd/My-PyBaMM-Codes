import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pybamm

# Create DFN model
model = pybamm.lithium_ion.DFN()

# Load the Chen2020 parameters
params = pybamm.ParameterValues("Chen2020")
#print("Loaded parameters:", params)
# Apply parameters to the model

'''
# Search for all parameters containing "electrolyte"
results = params.search("electrolyte")


sim = pybamm.Simulation(model, parameter_values=params)
sim.solve([0, 3600])  # Solve for 1 hour    
sim.plot()

#params.search("current")
params["Current function [A]"] 
print("Current function [A]:", params["Current function [A]"])
sim= pybamm.Simulation(model)
sim.solve([0, 3600])  # Solve for 1 hour
params["Current function [A]"] = 1.0  # Set a constant current of 1 A
new_sim= pybamm.Simulation(model, parameter_values=params)
new_sim.solve([0, 3600])  # Solve for 1 hour

solutions= [sim.solution, new_sim.solution]

for sol in solutions:
    d_cap = sol["Discharge capacity [A.h]"].data
    voltage = sol["Terminal voltage [V]"].data
    plt.plot(d_cap, voltage)
# try to plot the new and old solutions on the same plot

plt.xlabel("Discharge Capacity [A.h]")
plt.ylabel("Terminal Voltage [V]")
plt.title("Terminal Voltage vs Discharge Capacity")
plt.grid()
plt.legend(["Original Current", "Modified Current"])
plt.show()
'''

def my_current_function(t):
    """
    Custom current function that returns a constant current of Sin function.
    This can be modified to implement different current profiles.
    """
    return 1.0*pybamm.sin(2*np.pi*t/60) 

params["Current function [A]"] = my_current_function  # Set the custom current function


#t_eval = np.linspace(0, 3600, 100)  # Time evaluation points
#Sin_sim.solve(t_eval)  # Solve for 1 hour
#Sin_sim.plot()

# here i am changing the solver to use CasadiSolver
solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3)  # Set solver tolerances
Sin_sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
t_eval = np.linspace(0, 3600, 100) 
Sin_sim.solve(t_eval)
Sin_sim.plot()