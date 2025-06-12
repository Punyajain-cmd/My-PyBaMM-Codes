import pybamm
import numpy as np
models = pybamm.lithium_ion.SPM()
parameters = pybamm.ParameterValues("Chen2020")
sim = pybamm.Simulation(models, parameter_values=parameters)
t_eval = np.linspace(0, 3600, 100)
sim.solve(t_eval=t_eval)
sim.plot(["Terminal voltage [V]", "Current [A]", "Discharge capacity [A.h]"])

