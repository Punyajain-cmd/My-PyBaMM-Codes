import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pybamm

# Create and solve SPM model
model = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model)
solution = sim.solve([0, 3600])  # Time in seconds (1 hour)

# To search for variables containing "electrolyte"
# This is the correct usage
'''electrolyte_vars = [name for name in model.variables.keys() if "electrolyte" in name.lower()]
print("Electrolyte-related variables:")
for var in electrolyte_vars:
    print(" -", var)

# Plot selected variables
sim.plot([
    ["Electrolyte current density [A.m-2]", "Electrode current density [A.m-2]"],
    "Terminal voltage [V]"
])
'''
solution= sim.solution
# Print solution summary
print("Solution summary:")

V=solution["Terminal voltage [V]"]

#V(t=1800)  # Get voltage at 30 minutes (1800 seconds)

print("Voltage at 30 minutes:", V(t=1800))  

print("Full voltage data array:")
print(V.data) 

D=solution["Electrolyte current density [A.m-2]"]

print("Electrolyte current density [A.m-2] at 30 minutes:", D(t=1800))

print("Full Electrolyte current density [A.m-2] data array:")
print(D.data)
'''
solution.save_data("SPM_solution_data.h5")  # Save the solution data to an HDF5 file
solution.save_data("SPM_solution_data.csv,[Time [s], Terminal voltage [V]")

# Open the file in read mode
with h5py.File("SPM_solution_data.h5", "r") as f:
    def print_structure(name, obj):
        print(name)
        '''


discharge_capacity = solution["Discharge capacity [A.h]"].data
plt.plot(discharge_capacity, V.data)
plt.xlabel("Discharge Capacity [A.h]")
plt.ylabel("Terminal Voltage [V]")
plt.grid()
plt.title("Terminal Voltage vs Discharge Capacity")