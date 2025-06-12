import pybamm
import matplotlib.pyplot as plt
# Load the Single Particle Model
model = pybamm.lithium_ion.SPM()

# Create a simulation
sim = pybamm.Simulation(model)

# Solve the model
sim.solve([0, 3600])  # simulate for 1 hour

# Plot the results
sim.plot()
