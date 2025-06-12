import pybamm
import  numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import pybamm.solvers

'''
model = pybamm.BaseModel(name="base model")
y = pybamm.Variable("y")
a = pybamm.Parameter("a")
model.rhs = {y: a * y}
model.initial_conditions = {y: 1}
print(model.rhs[y])

params = pybamm.ParameterValues({"a" : 2})
params.process_model(model)
print(model.rhs[y])


solver = pybamm.CasadiSolver( mode="fast", rtol=1e-6, atol=1e-6)

t_eval = np.linspace(0, 1, 100)
solution = solver.solve(model, t_eval=t_eval)
plt.plot(solution.t, solution.y.T) # here we have taken transpose and this is necessary for the plot to work in matplotlib
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.title("Solution of the base model")
plt.grid(True)
plt.show()
'''

'''
# in the above code we created a base model using standard parameter
# now we will create a model using a custom parameter function, basically we will use input parameter function to define the model 

model = pybamm.BaseModel(name="base model with a input parameter function")
y = pybamm.Variable("y")
a = pybamm.InputParameter("a")  # Use InputParameter instead of Parameter

model.rhs = {y: a * y}
model.initial_conditions = {y: 1}

solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
t_eval = np.linspace(0, 1, 100)
for a_value in  [1, 2, 3, 4]:
    #params = pybamm.ParameterValues({"a": a_value})
    #params.process_model(model)
    solution = solver.solve(model, t_eval, inputs= {"a": a_value})
    plt.plot(solution.t, solution.y.T)
    label = 'a={}'.format(a_value)

# if the code written below was inside the loop then there will be a plot for each value of a
# now its outside the loop so there will be a plot for all the values of a in the same plot
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.title("Solution of the base model with the input parameter function")
plt.grid(True)
plt.legend()
plt.show()
'''
# this code is used to create a model with a custom parameter function

'''
model = pybamm.BaseModel(name="name")
y = pybamm.Variable("y")
a = pybamm.Parameter("a")
model.rhs = ({y: a*y})  
model.initial_conditions = {y: 1}
model.variables = ({"y_squared" : y**2}) # this is a custom variable that we are adding into the model

params = pybamm.ParameterValues({"a": '[input]'}) # Use a string to indicate its a input parameter
params.process_model(model)

solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
t_eval = np.linspace(0, 1, 100)


for a_values in [1, 2, 3, 4]:
    solution = solver.solve(model, t_eval, inputs={"a": a_values})
    plt.plot(solution.t, solution.y.T, label="a={}".format(a_values))

plt.xlabel("Time [s]")
plt.ylabel("y")
plt.grid(True)
plt.title("Solution of the base model with the custom parameter function")
plt.legend()
plt.show()
'''
# in this above code what we did is that used the standard parameter and then we used input parameter to change the value of the parameter

'''
model = pybamm.lithium_ion.DFN()
params = model.default_parameter_values
geometry = model.default_geometry
params.update({"Current function [A]" : "[input]"})  # Set a constant current function as an input parameter
params.process_model(model)


solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
t_eval = np.linspace(0, 3600, 100)  # Simulate for 1 hour


for current in [1, 2, 3, 4]:
    sim = pybamm.Simulation(model, parameter_values=params, geometry=geometry)
    # Set the current function as an input parameter
    current_function = pybamm.InputParameter("Current function [A]")
    params.update({"Current function [A]": current_function})
    solution = sim.solve(t_eval, inputs={"Current function [A]": current})
    plt.plot(solution.t, solution.y.T, label="Current = {} A".format(current))

plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("DFN Model with Custom Current Function")
plt.legend()
plt.grid(True)
plt.show()
'''
# In this code, we create a DFN model and set a custom current function as an input parameter.


data = solver.solve(model, t_eval, inputs={"Current function [A]": 0.2222})["Terminal voltage [V]"](t_eval) # this line of code allows to generate the data u