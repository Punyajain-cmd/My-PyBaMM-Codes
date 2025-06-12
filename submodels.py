import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pybamm


'''
dfn= pybamm.lithium_ion.DFN()
dfn.submodels #this show the submodels used in the DFN model
dfn.param = pybamm.ParameterValues("Chen2020")
#print(dfn.param)
#print(dfn.submodels)


# the best thing is that in pybamm, you can just change the submodels in an already build model to capture some new features or physics with other things remaining the same



spm= pybamm.lithium_ion.SPM()
spm.submodels #this show the submodels used in the SPM model
spm.submodels["electrolyte diffusion"]
print(spm.submodels["electrolyte diffusion"])
# the output says that the electrolyte diffusion submodel is not used (it has a constant value) in the SPM model, which is expected since SPM does not include electrolyte diffusion


dfn.submodels["electrolyte diffusion"]
print(dfn.submodels["electrolyte diffusion"])
# the output shows that the electrolyte diffusion submodel is used (at its full capacity, which means the complete diffusion equations for the electrolyte is being solved) in the DFN model, which is expected since DFN includes electrolyte diffusion
'''
# the above part just explain how to get different submodels present in a model, now we will see how to change the submodels in a model


'''
dfn = pybamm.lithium_ion.DFN(name="dfn")
tdfn = pybamm.lithium_ion.DFN(options={"thermal" : "lumped"}, name="tdfn")

dfn.submodels["electrolyte diffusion"]
print(dfn.submodels["electrolyte diffusion"])
tdfn.submodels["electrolyte diffusion"]
print(tdfn.submodels["electrolyte diffusion"])
                                                                    # the output for both the models is same as there is no changes made in the electrolyte diffusion part of 

dfn.submodels["thermal"]
print(dfn.submodels["thermal"])
tdfn.submodels["thermal"]
print(tdfn.submodels["thermal"])
                                                                    # the output we this case is going to be different for both the models as we changed the thermal submodel in one of the model

models=[dfn , tdfn]                                                 # we will now plot the submodels of both the models. so I just made a list of all the models being used
params = pybamm.ParameterValues("Chen2020")                         # using the same parameters for both the models
sols = []                                                           # to store the solutions of both the models

for model in models:
    sim = pybamm.Simulation(model, parameter_values=params)
    sim.solve([0, 3600])  # solve for 1 hour
    sols.append(sim.solution)

# Plot the submodels
pybamm.dynamic_plot(sols, ["Terminal voltage [V]", "Volume-averaged cell temperature [C]"])

'''

# above code shows how to change the submodels in a model and how to plot the submodels of both the models.



# Now we will see how to change the submodels in a model to include lithium platting 

model11 = pybamm.lithium_ion.DFN(options={"lithium plating": "reversible"}, name="model11")
model12 = pybamm.lithium_ion.DFN(options={"lithium plating": "irreversible"}, name="model12")

#parameter updates
parameter_values = pybamm.ParameterValues.Chen2020()  # Load the Chen2020 parameter values


parameter_values.update({"Current function [A]": 1, "Upper voltage cut-off [V]": 4.2, "Lithium plating kinetic rate constant [m.s-1]": 1e-9})
experiment_discharge = pybamm.Experiment(["discharge"])  # Define the experiment_discharge variable

sim_discharge1 = pybamm.Simulation(model11, parameter_values=parameter_values, experiment = experiment_discharge)
sol_discharge1 = sim_discharge1.solve([0, 3600])  # solve for 1 hour
model11.set_initial_conditions_from(sol_discharge1, inplace=True)  # set initial conditions from the discharge solution


sim_discharge2 = pybamm.Simulation(model12, parameter_values=parameter_values, experiment = experiment_discharge)
sol_discharge2 = sim_discharge2.solve([0, 3600])  # solve for 1 hour
model12.set_initial_conditions_from(sol_discharge2, inplace=True)  # set initial conditions from the discharge solution

experiment_2C = pybamm.Experiment([("charge at 2C until 4.2 V", "Hold at 4.2 V until C/20", "Rest for 1 hour")])
experiment_1C = pybamm.Experiment([("charge at 1C until 4.2 V", "Hold at 4.2 V until C/20", "Rest for 1 hour")])
experiment_cover2 = pybamm.Experiment([("charge at C/2 until 4.2 V", "Hold at 4.2 V until C/20", "Rest for 1 hour")])
# similarly many different experiments can be defined

sim_discharge1 = pybamm.Simulation(model11, parameter_values=parameter_values, experiment=(experiment_2C, experiment_1C, experiment_cover2))
sim_discharge2 = pybamm.Simulation(model12, parameter_values=parameter_values, experiment=(experiment_2C, experiment_1C, experiment_cover2))

# Plot the submodels
pybamm.dynamic_plot([sol_discharge1, sol_discharge2], ["Terminal voltage [V]", "Volume-averaged cell temperature [C]"], labels=["model11", "model12"])